# Complete Research Reproduction Guide
The repository contains the data files and code files needed to reach the conclusions in the article “Mechanistic Prediction and High-Throughput Risk Screening of Aquatic Contaminants Using *Daphnia magna* AOP Network”.

### The data files consist of 2 pieces:
- **allDescriptors.csv:** Feature matrix obtained after molecular structure optimization and descriptors calculation for all chemicals (6,971 chemicals ×10,024 structure descriptors).
- **domain_dir:** Processed bioactivity data for all 24 MIEs/KEs in the *Daphnia magna* AOP network.

### The code files include 9 scripts:
- *S1_features_preprocess.py*
- *S2_TBE_train_model.py*
- *S3_TBE_predict_external.py*
- *S4_TBE_ADplot.py*
- *S5_MLP_train_model.py*
- *S6_MLP_predict_external.py*
- *S7_MLP_ADplot.py*
- *S8_ROCplot.py*
- *S9_ADcoverage_plot.py*

Among these, *S5_MLP_train_model.py* is the core script for Domain-Cross Multi-Layer Perceptron (MLP) Model training. 

To reproduce this study, run the scripts in the order of S1-S9 and save the intermediate files, and pay attention to modifying the file path before running the scripts.
