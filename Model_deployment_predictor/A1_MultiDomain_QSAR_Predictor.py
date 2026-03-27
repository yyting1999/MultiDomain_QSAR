"""
Integrated tool for applicability domain assessment and multi-domain MLP model prediction
Functionality:
1. Load unknown sample features
2. Fill missing features using KNN
3. Calculate local density ratio and determine if within structural applicability domain
4. Predict multiple targets using trained multi-domain MLP model
5. Evaluate prediction reliability based on confidence threshold
6. Output complete results
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader
from S5_MLP_train_model import MultiDomainClassifier, MixedFeatureScaler, DomainIndexManager
import warnings
warnings.filterwarnings('ignore')


class MultiDomainQSARPredictor:
    """Optimized predictor with applicability domain assessment"""
    
    def __init__(self, model_dir='model_files', ad_data_dir='model_files/ad_data'):
        """Initialize predictor"""
        self.model_dir = Path(model_dir)
        self.ad_data_dir = Path(ad_data_dir)
        
        # Setup logging
        self._setup_logging()
        
        # Model related
        self.model = None
        self.scaler = None
        self.domain_manager = None
        self.selected_features = []
        self.domain_names = []
        
        # Applicability domain related
        self.global_train_features = None
        self.global_density_data = None
        self.density_threshold = None
        self.confidence_threshold = 0.525
        
        # KNN imputation related
        self.knn_imputer = None
        self.pca = None
        self.binary_features = []  # Binary feature indices
        self.continuous_features = []  # Continuous feature indices
        self.binary_feature_names = []  # Binary feature names
        self.feature_names = []  # All feature names
        self.global_defaults = {}  # Global default values

        # Training set original features (for KNN imputation)
        self.train_original_features = None
        self.train_original_features_df = None
    
    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('prediction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_required_files(self):
        """Load all required files"""
        self.logger.info("=" * 60)
        self.logger.info("Loading all required files")
        self.logger.info("=" * 60)
        
        try:
            # 1. Load model related files
            self.logger.info("1. Loading model related files...")
            
            # 1.1 Load full model
            model_path = self.model_dir / 'MLP_final_model_full.pth'
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = torch.load(model_path, map_location='cpu')
            self.model.eval()
            self.logger.info("  ✓ Model loaded successfully")

            # 1.2 Validate model structure
            self.logger.info("Validating model structure...")
            if self.model is not None:
                # Check if model has shared_extractor
                if not hasattr(self.model, 'shared_extractor'):
                    self.logger.error("Model missing shared_extractor attribute")
                    raise AttributeError("Model structure incomplete")
                
                # Check model parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                self.logger.info(f"Total model parameters: {total_params:,}")
                
                # Check for NaN in model weights
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any():
                        self.logger.warning(f"Model parameter {name} contains NaN")
                    if torch.isinf(param).any():
                        self.logger.warning(f"Model parameter {name} contains infinite values")   
            
            # 1.3 Load domain manager
            domain_path = self.model_dir / 'domain_manager.pt'
            if not domain_path.exists():
                raise FileNotFoundError(f"Domain manager file not found: {domain_path}")
            self.domain_manager = torch.load(domain_path)
            self.domain_names = self.domain_manager.domain_names
            self.logger.info(f"  ✓ Domain manager loaded successfully: {len(self.domain_names)} domains")
            
            # 1.4 Load feature metadata
            meta_path = self.model_dir / 'feature_meta.pt'
            if not meta_path.exists():
                # Try to find in ad_data directory
                alt_path = self.ad_data_dir / f'{self.domain_names[0]}_feature_meta.pt'
                if alt_path.exists():
                    meta_path = alt_path
                else:
                    raise FileNotFoundError("Feature metadata not found")
            
            feature_meta = torch.load(meta_path)
            self.selected_features = feature_meta['selected_features']
            self.categorical_indices = feature_meta.get('categorical_indices', [])
            self.logger.info(f"  ✓ Feature metadata loaded successfully: {len(self.selected_features)} features")
            self.logger.info(f"  ✓ Categorical feature indices: {len(self.categorical_indices)}")
            
            # 1.5 Load normalizer
            scaler_path = self.model_dir / 'scaler.pkl'
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            self.logger.info("  ✓ Scaler loaded successfully")
            
            # 2. Load applicability domain related files
            self.logger.info("2. Loading applicability domain related files...")
            
            # 2.1 Global training set features
            global_train_path = self.ad_data_dir / 'global_X_train.pt'
            if not global_train_path.exists():
                raise FileNotFoundError(f"Global training set file not found: {global_train_path}")
            
            global_train_data = torch.load(global_train_path)
            self.global_train_features = global_train_data['features']
            
            if isinstance(self.global_train_features, torch.Tensor):
                self.global_train_features = self.global_train_features.numpy()
            
            self.logger.info(f"  ✓ Global training set loaded successfully: {self.global_train_features.shape}")
            
            # 2.2 Global density data
            density_path = self.ad_data_dir / 'global_density_data.pt'
            if not density_path.exists():
                raise FileNotFoundError(f"Global density data file not found: {density_path}")
            self.global_density_data = torch.load(density_path)
            self.logger.info(f"  ✓ Global density data loaded successfully: {len(self.global_density_data)} samples")
            
            # 2.3 Density ratio threshold
            threshold_path = self.ad_data_dir / 'global_ad_threshold.json'
            if not threshold_path.exists():
                raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
            
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                self.density_threshold = threshold_data['threshold']
            
            self.logger.info(f"  ✓ Applicability domain threshold loaded successfully: {self.density_threshold:.6f}")
            
            # 2.4 Load training set original features for KNN imputation
            self.logger.info("Loading training set original features for KNN imputation...")
            train_original_path = self.model_dir / 'allDescriptors_filled.csv'
            if not train_original_path.exists():
                # Try to find in ad_data directory
                train_original_path = self.ad_data_dir / 'allDescriptors_filled.csv'

            if not train_original_path.exists():
                self.logger.warning("  ⚠ Training set original features file not found, KNN imputation will not be available")
                self.train_original_features = None
            else:
                # Read training set original features
                train_df = pd.read_csv(train_original_path)
                self.logger.info(f"  Training set original features shape: {train_df.shape}")
                
                # Select only features that match selected_features
                available_features = []
                for feat in self.selected_features:
                    if feat in train_df.columns:
                        available_features.append(feat)
                    else:
                        self.logger.warning(f"  Feature {feat} not found in training set original features")
                
                if len(available_features) < len(self.selected_features):
                    self.logger.warning(f"  Warning: Only {len(available_features)}/{len(self.selected_features)} features found in training set")
                    # Update selected_features to available features
                    self.selected_features = available_features
                
                # Load selected features
                self.train_original_features = train_df[self.selected_features].values
                self.train_original_features_df = train_df[self.selected_features].copy()
                self.logger.info(f"  ✓ Training set original features loaded successfully: {self.train_original_features.shape}")
            
            # 3. Initialize KNN imputer
            self.logger.info("3. Initializing KNN imputer...")
            self._init_knn_imputer()
            
            self.logger.info("=" * 60)
            self.logger.info("All files loaded successfully")
            self.logger.info("=" * 60)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load required files: {str(e)}")
            raise
    
    def _init_knn_imputer(self):
        """Initialize smart KNN imputer including binary feature identification"""
        if self.train_original_features is None:
            self.logger.warning("Cannot initialize KNN imputer: no training set original features")
            return
        
        try:
            # Identify binary features
            self._identify_binary_features()
            
            # Precompute global defaults for each feature
            self._compute_global_defaults()
            
            # Initialize transformers only (don't compute nearest neighbors between training samples)
            self._init_transformers_only()
            
            self.logger.info(f"  ✓ KNN imputer initialized successfully")
            self.logger.info(f"  ✓ Identified {len(self.binary_features)} binary features")
            self.logger.info(f"  ✓ Identified {len(self.continuous_features)} continuous features")
            
            if self.binary_feature_names:
                self.logger.info(f"  ✓ Binary feature examples: {self.binary_feature_names[:3]}")
            
        except Exception as e:
            self.logger.warning(f"KNN imputer initialization failed: {str(e)}, will use default imputation")
            self.temp_imputer = None
            self.scaler_pca = None
            self.pca = None
            self.binary_features = []
            self.continuous_features = []

    def _init_transformers_only(self):
        """Initialize transformers only, do not compute nearest neighbors between training samples"""
        from sklearn.impute import SimpleImputer
        
        # 1. Create temporary imputer
        self.logger.info("  Initializing temporary imputer...")
        self.temp_imputer = SimpleImputer(strategy='median')
        X_temp = self.temp_imputer.fit_transform(self.train_original_features)
        
        # 2. Initialize standardizer
        self.logger.info("  Initializing standardizer...")
        self.scaler_pca = StandardScaler()
        X_scaled = self.scaler_pca.fit_transform(X_temp)
        
        # 3. Initialize PCA
        self.logger.info("  Initializing PCA...")
        self.pca = PCA(n_components=100, random_state=42)
        self.pca.fit(X_scaled)
        
        self.logger.info("  Transformers initialized successfully")

    def _identify_binary_features(self):
        """Identify binary features (0/1) - based on training set original features"""
        self.logger.info("Identifying binary features...")
        
        binary_features = []
        continuous_features = []
        binary_feature_names = []
        
        for i, feat_name in enumerate(self.selected_features):
            if i < self.train_original_features.shape[1]:
                col_values = self.train_original_features[:, i]
                
                # Get unique non-NaN values
                unique_vals = np.unique(col_values[~np.isnan(col_values)])
                
                if len(unique_vals) > 0:
                    # Check if binary feature (0/1)
                    is_binary = (set(unique_vals) <= {0, 1}) and (0 in unique_vals or 1 in unique_vals)
                    
                    if is_binary:
                        binary_features.append(i)
                        binary_feature_names.append(feat_name)
                    else:
                        continuous_features.append(i)
                else:
                    # If all values are NaN, treat as continuous feature
                    continuous_features.append(i)
            else:
                # Feature index out of range, treat as continuous feature
                continuous_features.append(i)
        
        self.binary_features = binary_features
        self.continuous_features = continuous_features
        self.binary_feature_names = binary_feature_names
        
        return binary_features, continuous_features
    
    def _compute_global_defaults(self):
        """Compute global default values for each feature (based on training set)"""
        self.logger.info("Computing global default values...")
        
        for i, feat_name in enumerate(self.selected_features):
            if i < self.train_original_features.shape[1]:
                col_values = self.train_original_features[:, i]
                valid_values = col_values[~np.isnan(col_values)]
                
                if len(valid_values) > 0:
                    if i in self.binary_features:
                        # Binary feature: use mode
                        unique_values, counts = np.unique(valid_values, return_counts=True)
                        if len(unique_values) > 0:
                            self.global_defaults[feat_name] = unique_values[np.argmax(counts)]
                        else:
                            self.global_defaults[feat_name] = 0
                    else:
                        # Continuous feature: use median
                        self.global_defaults[feat_name] = np.median(valid_values)
                else:
                    # If no valid values, use 0
                    self.global_defaults[feat_name] = 0
        
        self.logger.info(f"  Computed global defaults for {len(self.global_defaults)} features")

    def load_unknown_samples(self, feature_file, sample_id_col='Compound_CID'):
        """Load unknown sample feature file"""
        self.logger.info("=" * 60)
        self.logger.info(f"Loading unknown samples: {feature_file}")
        
        try:
            # Read file
            if feature_file.endswith('.csv'):
                df = pd.read_csv(feature_file)
            elif feature_file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(feature_file, engine='openpyxl')
            else:
                raise ValueError("Only CSV or Excel formats are supported")
            
            self.logger.info(f"Original data shape: {df.shape}")
            
            # Detect sample ID column
            if sample_id_col not in df.columns:
                for col in ['CID', 'Compound_ID', 'ID', 'Name', 'SMILES', 'compound_id']:
                    if col in df.columns:
                        sample_id_col = col
                        self.logger.info(f"Auto-detected sample ID column: {col}")
                        break
                else:
                    raise ValueError(f"Sample ID column not found, please ensure file contains '{sample_id_col}' column")
            
            # Process sample IDs
            df = df.copy()
            df[sample_id_col] = df[sample_id_col].astype(str).str.strip()
            sample_ids = df[sample_id_col].tolist()
            
            # Set index
            df_indexed = df.set_index(sample_id_col)
            
            self.logger.info(f"Successfully loaded {len(sample_ids)} samples")
            return df_indexed, sample_ids
            
        except Exception as e:
            self.logger.error(f"Failed to load feature file: {str(e)}")
            raise
    
    def process_features(self, features_df, sample_ids):
        """
        Process features: fill missing values, normalize, extract shared features
        
        Returns:
        - normalized_features: Normalized original features (for model prediction)
        - shared_features: Shared features (128-dim, for density ratio calculation)
        - processed_df: Processed feature DataFrame
        """
        self.logger.info("=" * 60)
        self.logger.info("Processing features...")
        
        try:
            # 1. Check and process features
            missing_features = set(self.selected_features) - set(features_df.columns)
            extra_features = set(features_df.columns) - set(self.selected_features)
            
            # If missing feature columns, directly report error instead of filling with 0
            if missing_features:
                error_msg = f"Input data missing {len(missing_features)} required feature columns.\n"
                error_msg += f"Missing feature columns: {list(missing_features)[:10]}\n"
                if len(missing_features) > 10:
                    error_msg += f"... and {len(missing_features)-10} other features\n"
                error_msg += "\nSolution:\n"
                error_msg += "1. Ensure input file contains all required feature columns\n"
                error_msg += "2. If a feature has no value, keep the column name and set value to NaN\n"
                error_msg += "3. System will automatically fill NaN values using KNN method\n"
                error_msg += f"\nComplete feature list saved in: model_required_features.csv"
                
                self.logger.error(error_msg)
                raise ValueError(f"Input data missing {len(missing_features)} required feature columns")
            
            if extra_features:
                self.logger.info(f"Ignoring {len(extra_features)} extra feature columns")
            
            # 2. Select features in selected_features order
            selected_df = features_df[self.selected_features].copy()
            self.logger.info(f"Feature selection completed: {selected_df.shape}")
            
            # 3. Check for NaN values in features, use KNN imputation
            nan_before_norm = selected_df.isna().sum().sum()
            if nan_before_norm > 0:
                self.logger.info(f"Found {nan_before_norm} NaN values, using KNN imputation...")
                selected_df = self._knn_fill_missing_values_train_style(selected_df)
            else:
                self.logger.info("No NaN values found, no imputation needed")
            
            # 4. Feature normalization
            if self.scaler is not None:
                self.logger.info("Applying normalizer...")
                
                try:
                    scaled_features = self.scaler.transform(selected_df.values)
                    self.logger.info("Feature normalization completed")
                except Exception as e:
                    self.logger.error(f"Normalization failed: {str(e)}")
                    scaled_features = selected_df.values
                    self.logger.warning("Using unnormalized features")
            else:
                scaled_features = selected_df.values
                self.logger.warning("Normalizer not used")
            
            # 5. Check normalized features
            nan_after_norm = np.sum(np.isnan(scaled_features))
            if nan_after_norm > 0:
                self.logger.warning(f"Normalized features contain {nan_after_norm} NaN values")
                scaled_features = np.nan_to_num(scaled_features, nan=0.0)
                self.logger.warning("Filled NaN in normalized features with 0")
            
            # 6. Convert to Tensor
            features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
            self.logger.info(f"Feature tensor shape: {features_tensor.shape}")
            
            # 7. Extract shared features (128-dim)
            with torch.no_grad():
                self.model.eval()
                
                # Extract shared features
                try:
                    shared_features = self.model.shared_extractor(features_tensor)
                    shared_features_np = shared_features.cpu().numpy()
                    
                    # Record shared feature statistics
                    self.logger.info(f"Shared feature extraction completed")
                    self.logger.info(f"Shared feature shape: {shared_features_np.shape}")
                    self.logger.info(f"Shared feature statistics:")
                    self.logger.info(f"  Minimum: {np.min(shared_features_np):.6f}")
                    self.logger.info(f"  Maximum: {np.max(shared_features_np):.6f}")
                    self.logger.info(f"  Mean: {np.mean(shared_features_np):.6f}")
                    self.logger.info(f"  Std: {np.std(shared_features_np):.6f}")
                    
                except Exception as e:
                    self.logger.error(f"Shared feature extraction failed: {str(e)}")
                    raise
            
            return features_tensor, selected_df, shared_features_np
            
        except Exception as e:
            self.logger.error(f"Feature processing failed: {str(e)}")
            raise
    
    def _knn_fill_missing_values_train_style(self, query_df):
        """Use training-time method to KNN impute missing values in query data"""
        self.logger.info("Using training-time method to KNN impute missing values in query data...")
        
        if self.train_original_features is None or self.pca is None or self.temp_imputer is None or self.scaler_pca is None:
            self.logger.warning("KNN imputer not initialized, using global default imputation")
            return self._fill_with_global_defaults_for_nan(query_df)
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Process query data
            query_data = query_df.values.copy()
            
            # 1. Use temporary imputer (median) for query samples
            query_data_filled = self.temp_imputer.transform(query_data)
            
            # 2. Standardization
            query_scaled = self.scaler_pca.transform(query_data_filled)
            
            # 3. PCA dimensionality reduction
            query_pca = self.pca.transform(query_scaled)
            
            # 4. Calculate PCA representation of training set
            train_temp = self.temp_imputer.transform(self.train_original_features)
            train_scaled = self.scaler_pca.transform(train_temp)
            train_pca = self.pca.transform(train_scaled)
            
            # 5. Calculate nearest neighbors for each query sample in training set PCA space
            knn = NearestNeighbors(n_neighbors=20, n_jobs=-1)
            knn.fit(train_pca)
            
            # Get neighbor indices for each query sample (indices in training set)
            _, neighbor_indices = knn.kneighbors(query_pca)
            
            # 6. Impute missing values
            X_filled = query_data.copy()
            
            for i in range(len(query_df)):
                current_sample = query_df.iloc[i]
                missing_mask = current_sample.isna()
                missing_features = missing_mask[missing_mask].index.tolist()
                
                if not missing_features:
                    continue
                
                # Get current sample's neighbor indices (indices in training set)
                neighbors = neighbor_indices[i]
                
                for feature in missing_features:
                    if feature not in self.selected_features:
                        continue
                    
                    feat_idx = self.selected_features.index(feature)
                    
                    # Extract feature values from training set
                    neighbor_values = self.train_original_features[neighbors, feat_idx]
                    # Ensure no NaN
                    neighbor_values = neighbor_values[~np.isnan(neighbor_values)]
                    
                    if len(neighbor_values) > 0:
                        if feat_idx in self.binary_features:
                            # Binary feature processing: use neighbor mode
                            values, counts = np.unique(neighbor_values, return_counts=True)
                            most_common_value = values[np.argmax(counts)]
                            fill_value = int(most_common_value)
                        else:
                            # Continuous feature: use neighbor mean
                            fill_value = np.mean(neighbor_values)
                    else:
                        # Use global default if all neighbors are missing
                        fill_value = self.global_defaults.get(feature, 0)
                    
                    X_filled[i, feat_idx] = fill_value
            
            # Update DataFrame
            for i, col in enumerate(self.selected_features):
                if col in query_df.columns and query_df[col].isna().any():
                    query_df[col] = X_filled[:, i]
            
            self.logger.info("KNN imputation completed")
            return query_df
            
        except Exception as e:
            self.logger.warning(f"KNN imputation failed: {str(e)}, falling back to global default imputation")
            return self._fill_with_global_defaults_for_nan(query_df)
    
    def _fill_with_global_defaults_for_nan(self, query_df):
        """Use global default values to impute NaN values"""
        self.logger.info("Using global default values to impute NaN values...")
        
        for col in query_df.columns:
            if query_df[col].isna().any():
                if col in self.selected_features:
                    default_value = self.global_defaults.get(col, 0)
                    query_df[col] = query_df[col].fillna(default_value)
                    self.logger.debug(f"Feature {col} filled with global default {default_value}")
                else:
                    query_df[col] = query_df[col].fillna(0)
                    self.logger.warning(f"Feature {col} not in selected_features, filled with 0")
        
        return query_df
    
    def compute_density_ratios_ad(self, query_shared_features):
        """Calculate local density ratio using shared features"""
        self.logger.info("=" * 60)
        self.logger.info("Calculating local density ratio (using shared features)...")
        
        try:
            if self.global_train_features is None or self.global_density_data is None:
                raise ValueError("Global training set or density data not loaded")
            
            # Create data copies
            query_features_clean = query_shared_features.copy()
            global_features_clean = self.global_train_features.copy()
            density_data_clean = self.global_density_data.copy()
            
            # Calculate distance matrix
            self.logger.info("Calculating distance matrix...")
            
            try:
                dist_matrix = pairwise_distances(
                    query_features_clean,
                    global_features_clean
                )
                self.logger.info(f"Distance matrix calculation completed: shape={dist_matrix.shape}")
                
            except Exception as e:
                self.logger.error(f"Distance matrix calculation failed: {str(e)}")
                n_query = len(query_features_clean)
                n_global = len(global_features_clean)
                dist_matrix = np.ones((n_query, n_global))
                self.logger.warning("Using unit distance matrix as fallback")
            
            # Calculate density ratios
            self.logger.info("Calculating density ratios...")
            density_ratios = []
            
            for i in range(len(query_features_clean)):
                # Find nearest neighbor
                try:
                    nn_idx = np.argmin(dist_matrix[i])
                    d_query = dist_matrix[i][nn_idx]
                    d_nn = density_data_clean[nn_idx]
                    
                    # Calculate density ratio
                    if d_nn > 0:
                        density_ratio = d_query / d_nn
                    else:
                        # If d_nn is 0, use a small positive number
                        density_ratio = d_query / 1e-10
                        self.logger.debug(f"Sample {i} d_nn={d_nn}, using small value to avoid division by zero")
                    
                    density_ratios.append(density_ratio)
                    
                except Exception as e:
                    self.logger.warning(f"Sample {i} density ratio calculation failed: {str(e)}, using default value 1.0")
                    density_ratios.append(1.0)
            
            density_ratios = np.array(density_ratios)
            
            # Handle abnormal density ratios
            inf_mask = np.isinf(density_ratios)
            if np.any(inf_mask):
                self.logger.warning(f"Found {np.sum(inf_mask)} infinite density ratios")
                finite_ratios = density_ratios[~inf_mask]
                if len(finite_ratios) > 0:
                    max_finite = np.max(finite_ratios)
                    density_ratios[inf_mask] = max_finite + 1.0
                else:
                    density_ratios[inf_mask] = 5.0
            
            # Determine if within structural applicability domain
            # Use saved threshold
            in_ad_density = density_ratios <= self.density_threshold
            
            self.logger.info(f"Density ratio calculation completed")
            self.logger.info(f"Density ratio range: [{np.min(density_ratios):.4f}, {np.max(density_ratios):.4f}]")
            self.logger.info(f"Applicability domain threshold: {self.density_threshold:.4f}")
            self.logger.info(f"Samples within structural applicability domain: {np.sum(in_ad_density)}/{len(in_ad_density)}")
            
            return density_ratios, in_ad_density
            
        except Exception as e:
            self.logger.error(f"Density ratio calculation failed: {str(e)}")
            raise
  
    def predict_with_model(self, features_tensor, sample_ids, batch_size=128):
        """Predict using multi-domain MLP model"""
        self.logger.info("=" * 60)
        self.logger.info("Predicting using multi-domain MLP model...")
        
        try:
            dataset = TensorDataset(features_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize result storage
            results = {
                'sample_ids': sample_ids,
                'predictions': {domain: [] for domain in self.domain_names},
                'probabilities': {domain: [] for domain in self.domain_names},
                'confidences': {domain: [] for domain in self.domain_names}
            }
            
            total_batches = len(dataloader)
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader, 1):
                    inputs = batch[0]
                    
                    for domain_idx, domain_name in enumerate(self.domain_names):
                        # Set current domain
                        domain_tensor = torch.full(
                            (inputs.size(0),), 
                            domain_idx, 
                            dtype=torch.long
                        )
                        
                        # Predict
                        outputs = self.model(inputs, domain_tensor)
                        probs = outputs.cpu().numpy().flatten()
                        
                        # Get prediction class and confidence
                        preds = (probs > 0.5).astype(int)
                        confidences = np.maximum(probs, 1 - probs)
                        
                        results['predictions'][domain_name].extend(preds.tolist())
                        results['probabilities'][domain_name].extend(probs.tolist())
                        results['confidences'][domain_name].extend(confidences.tolist())
                    
                    # Progress display
                    if batch_idx % 10 == 0 or batch_idx == total_batches:
                        self.logger.info(f"  Processing progress: {batch_idx}/{total_batches} batches")
            
            self.logger.info("Model prediction completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Model prediction failed: {str(e)}")
            raise
    
    def evaluate_applicability_domain(self, confidences_dict, in_ad_density):
        """Evaluate applicability domain (confidence and combined applicability domain)"""
        self.logger.info("=" * 60)
        self.logger.info("Evaluating applicability domain...")
        
        try:
            sample_count = len(next(iter(confidences_dict.values())))
            
            # Initialize result storage
            ad_results = {
                'in_ad_confidence': {domain: [] for domain in self.domain_names},
                'in_ad_combined': {domain: [] for domain in self.domain_names}
            }
            
            for domain in self.domain_names:
                confidences = np.array(confidences_dict[domain])
                
                # Confidence applicability domain
                in_ad_conf = confidences >= self.confidence_threshold
                ad_results['in_ad_confidence'][domain] = in_ad_conf
                
                # Combined applicability domain (satisfying both density ratio and confidence)
                in_ad_combined = in_ad_density & in_ad_conf
                ad_results['in_ad_combined'][domain] = in_ad_combined
            
            self.logger.info("Applicability domain evaluation completed")
            return ad_results
            
        except Exception as e:
            self.logger.error(f"Applicability domain evaluation failed: {str(e)}")
            raise
    
    def save_complete_results(self, sample_ids, density_ratios, in_ad_density, 
                             model_results, ad_results, output_file='multidomainQSAR_predictions.csv'):
        """Save complete results to CSV file"""
        self.logger.info("=" * 60)
        self.logger.info(f"Saving complete results to: {output_file}")
        
        try:
            # Create main DataFrame
            df_results = pd.DataFrame(index=sample_ids)
            df_results.index.name = 'Compound_CID'
            # Add applicability domain information
            df_results['density_ratio'] = density_ratios
            df_results['in_ad_density'] = in_ad_density
            
            # Add prediction information for each domain
            for domain in self.domain_names:
                df_results[f'{domain}_prediction'] = model_results['predictions'][domain]
                df_results[f'{domain}_probability'] = model_results['probabilities'][domain]
                df_results[f'{domain}_confidence'] = model_results['confidences'][domain]
                df_results[f'{domain}_in_ad_confidence'] = ad_results['in_ad_confidence'][domain]
                df_results[f'{domain}_in_ad_combined'] = ad_results['in_ad_combined'][domain]
            
            # Save CSV file
            df_results.to_csv(output_file)
            
            # Show first few rows preview
            self.logger.info("\nResults preview:")
            self.logger.info(df_results.head().to_string())
            
            self.logger.info(f"Complete results saved to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-domain MLP prediction tool (with applicability domain assessment)')
    parser.add_argument('--feature_file', type=str, required=True,
                       help='Feature file path (CSV/Excel format)')
    parser.add_argument('--sample_id_col', type=str, default='Compound_CID',
                       help='Sample ID column name (default: Compound_CID)')
    parser.add_argument('--model_dir', type=str, default='model_files',
                       help='Model file directory (default: model_files)')
    parser.add_argument('--ad_data_dir', type=str, default='model_files/ad_data',
                       help='Applicability domain data directory (default: MLPresults/ad_data)')
    parser.add_argument('--output', type=str, default='multidomainQSAR_predictions.csv',
                       help='Output file path (default: multidomainQSAR_predictions.csv)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    
    args = parser.parse_args()
    
    # Create predictor instance
    predictor = MultiDomainQSARPredictor(
        model_dir=args.model_dir,
        ad_data_dir=args.ad_data_dir
    )
    
    try:
        # 1. Load all required files
        predictor.load_required_files()
        
        # 2. Load unknown samples
        features_df, sample_ids = predictor.load_unknown_samples(
            args.feature_file, args.sample_id_col
        )
        
        # 3. Process features (fill missing values, normalize, extract shared features)
        features_tensor, processed_df, shared_features = predictor.process_features(features_df, sample_ids)
        
        # 4. Calculate local density ratio (using shared features)
        density_ratios, in_ad_density = predictor.compute_density_ratios_ad(shared_features)
        
        # 5. Predict using model (using normalized original features)
        model_results = predictor.predict_with_model(
            features_tensor, sample_ids, args.batch_size
        )
        
        # 6. Evaluate applicability domain
        ad_results = predictor.evaluate_applicability_domain(
            model_results['confidences'], in_ad_density
        )
        
        # 7. Save complete results
        predictor.save_complete_results(
            sample_ids, density_ratios, in_ad_density,
            model_results, ad_results, args.output
        )
        
        predictor.logger.info("=" * 60)
        predictor.logger.info("Prediction task completed!")
        predictor.logger.info("=" * 60)
        
    except Exception as e:
        predictor.logger.error(f"Prediction process failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
