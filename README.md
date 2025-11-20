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
  Contains scripts to generate SHAP explanations and permutation feature importance for a trained model and dataset.
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
  Contains scripts to generate RAC descriptors using molsplify and geometric descriptors using Zeo++.
- **`featurzion_demo.ipynb`**
  Demonstrates the featurization of crystal structures of MOFs to create two csv files that contain RAC descriptors and geometric descriptors respectively.

### 5. `models`
This folder contains all saved machine learning models produced during the project, including baseline models and optimized models, as well as scripts related to calculating statistical metrics of model predictions made on a dataset. 
- various **`.pkl`**
  files that each contain a saved machine learning model and can be directly loaded.
- **`testing_metrics.py`** 
  Contains scripts to calculate statistical metrics of a model's prediction of the target (band gap) for samples in a dataset.
- **`testing_metrics_demo.ipynb`** 
  Demonstrates the evaluation of different models by calculating the statistical metrics of their predictions of the target (band gap) for samples in the testing dataset.

### 6. `models_training`
This folder contains scripts related to training machine-learning models.
- **`models_training.py`**
  Contains scripts to fit different models on the training dataset and search for the set of hyperparameters that results in the most accurate model on the training dataset.
- **`models_training_demo.ipynb`** 
  Demonstrates the training of baseline models with a single fit and optimized models with hyperparameter optimization using grid search and 3-fold cross-validation.


