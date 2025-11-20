# CHE1147-Group11-Project
Machine learning of metal-organic framework design for predicting band gap energies.

## Usage
To use this repository, install all required Python packages from requirements.txt.
For featurization, this project uses Zeo++, which requires Cygwin if you are using Windows.
Each folder contains a Jupyter Notebook demonstrating how to use the associated scripts.

## Folder Structure
### 1. `EDA_dataprocessing`
This folder contains all scripts and notebooks used for data preprocessing before training the machine-learning models.
- **`DataMerger.py`** 
  Merges RAC, Zeo++ and QMOF label (band gap) CSV files into a single dataset.
- **`DataPrepDemo.ipynb`** 
  Demonstrates the merging of datasets, basic exploratory data analysis (EDA) and dataset splitting.
- **`DataPreparation.py`**
  Takes the merged raw dataset as input, cleans and reduces features, performs the train/test split.
- **`EDA_visualization.ipynb`**
  Contains visualizations used to explore and summarize the dataset (histograms, correlations, feature distributions, etc.).

### 2. `XAI`
This folder contains scripts related to model explainability.
- **`XAI.py`**
  Contains scripts to generates SHAP explanations and permutation feature importance for a trained model and dataset.
- **`XAI_demo.ipynb`**
  Demonstrates the use of functions to generate SHAP explanations and permutation feature importance.

### 3. `data`
This folder contains all data required to run the project.
- **`Raw data folder`**
  Contains the unprocessed outputs from featurization QMOF steps.
- **`merged_rac_zeo_bandgap.csv`**
 Final merged dataset containing the target label (band gap), geometric features, and RAC features.
- **`test.csv`**
  Cleaned and processed test split used for model evaluation.
- **`train.csv`**
  Cleaned and processed training split used for model development.

### 4. `featurization`
This folder contains scripts related to featurization of crystal structures of MOFs from cif files. 
- **`cif`**
  A folder that contains demo cif file representing the crystal structures of MOFs.
- **`featurzier.py`** 
  Contains scripts to generates RAC descriptors using molsplify and geometric descriptors using Zeo++.
- **`featurzion_demo.py`**
  Demonstrates the featurization of crystal structures of MOFs to create two csv files that contain RAC descriptors and geometric descriptors respectively.

### 5. `models`
This folder contains all saved machine-learning models produced during the project, including both default (untuned) versions and hyperparameter-optimized versions. These .pkl files can be loaded directly

### 6. `models_training`
This folder contains all scripts used for training machine-learning models, performing hyperparameter tuning, and evaluating baseline performance.
- **`models_training.py`**

get_regression_metrics() Returns a dictionary containing: MAE, MSE, R² ,MAPE, Max Error Used for evaluating all trained models.

dummy_regressor_baseline() Creates simple baseline predictors (mean and median). Useful for checking whether ML models actually learn meaningful patterns.

get_linear_model() Returns a simple linear regression model. 

get_default_krr() Kernel Ridge Regression model with RBF kernel. 

get_default_xgb() Default XGBoost regressor. 

get_default_rf() Default Random Forest regressor.

reg_cv() Function returns the best model found by cross-validation along with its best hyperparameters


