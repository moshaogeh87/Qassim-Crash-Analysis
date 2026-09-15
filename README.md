**Qassim Crash Severity Analysis**

**Overview**

This repository contains the custom Python code associated with the research article "Severity Prediction of Traffic Accidents With Recent Machine Learning Paradigms and SHAP."

The code implements the crash injury-severity analysis workflow using traffic crash data obtained from the Ministry of Transport (MOT), Saudi Arabia, for the Qassim Region. The workflow includes data cleaning and preprocessing, categorical-variable encoding, stratified training-test splitting, SMOTE-based class balancing, Bayesian hyperparameter optimization, Logistic Regression, Random Forest, and XGBoost classification, independent test-set evaluation, ROC/AUC analysis, XGBoost feature-importance analysis, and SHAP-based model interpretation and visualization.

**Repository Contents**

Final_Custom_code_Qassim_Crash_Analysis.py — main Python analysis script.

README.md — repository documentation and instructions for running the analysis.

requirements.txt — required third-party Python packages.

**Data Availability**

The crash dataset used in this research was obtained from the Ministry of Transport (MOT), Saudi Arabia, with authorization for research purposes. The authors do not own the dataset and are therefore not permitted to redistribute it or deposit it in a public open-access repository.

Researchers interested in obtaining the underlying data should submit a request through the appropriate official Saudi government channels:

https://www.rga.gov.sa

https://open.data.gov.sa/en/publishers/687e070e-ebbd-4803-b1cd-e56a25887e77

The dataset is not included in this GitHub repository.

Expected Input File

The analysis script expects the authorized input dataset to be named:

Qassim_Crash_Data.csv

For execution, place the authorized dataset in the same directory as Final_Custom_code_Qassim_Crash_Analysis.py.

Data Dictionary

**Target Variable**

The target variable is Accident Severity Category, encoded as:

Non-Fatal Injury = 0

Fatal Injury = 1

**Explanatory Variables**

The following explanatory variables are used to predict crash injury severity:

Lighting Conditions

Period of the Day

Weekday

Road Type

Road Status

Weather Status

Road Alignment Details

Damage Type

Accident Type

Vehicle Type

Accident Cause

Number of Vehicles Involved

**Analytical Workflow**

The script performs the following main steps:

Imports the required libraries and configures the analysis settings.

Loads the crash dataset.

Removes duplicate records and records containing missing values.

Defines the target and explanatory variables.

Performs a stratified 70% training and 30% testing split.

Fits the categorical-variable preprocessing procedure on the training data and transforms the training and test datasets.

Applies SMOTE to the encoded training data.

Defines the hyperparameter search spaces for Logistic Regression, Random Forest, and XGBoost.

Performs Bayesian hyperparameter optimization using cross-validated ROC-AUC.

Retrieves the optimized model parameters.

Constructs and fits the final Logistic Regression, Random Forest, and XGBoost models.

Evaluates the models using the independent test dataset.

Calculates Accuracy, Precision, Recall, F1-score, and ROC-AUC.

Generates confusion matrices.

Generates ROC curves and compares AUC values.

Calculates and visualizes XGBoost feature importance.

Performs SHAP-based interpretation of the XGBoost model.

Generates global and class-specific SHAP analyses.

Generates SHAP main-effect plots for Accident Type and Weather Status.

Saves the analytical results, metadata, tables, and figures.
**
**Installation****

Python 3 is required. Install the required third-party packages using:

pip install -r requirements.txt

Running the Analysis

Download or clone this repository.

Obtain authorized access to the crash dataset.

Name the data file Qassim_Crash_Data.csv.

Place it in the same directory as Final_Custom_code_Qassim_Crash_Analysis.py.

Install the required packages.

Run:

python Final_Custom_code_Qassim_Crash_Analysis.py

The script automatically creates an analysis_outputs directory for the generated results.

**Generated Outputs**

Depending on successful execution, the script generates outputs including:

data-cleaning summary;

Bayesian hyperparameter-search results;

optimized model parameters;

model-performance metrics;

confusion matrices;

ROC/AUC comparison;

XGBoost feature-importance results and figure;

global SHAP feature-importance results;

fatal and non-fatal injury SHAP summaries;

class-conditional SHAP importance;

SHAP main-effect plots for Accident Type and Weather Status; and

analysis metadata.

**Reproducibility**

A fixed random state (random_state = 42) is used throughout the analysis where applicable to support reproducible execution. The analysis uses a stratified 70/30 training-test split and five-fold cross-validation during hyperparameter optimization.

Full numerical reproduction of the published analysis requires authorized access to the underlying Qassim_Crash_Data.csv dataset. Because the dataset cannot be publicly redistributed by the authors, it is not included in this repository.

**Citation**

If you use this code, please cite the associated research article:

Severity Prediction of Traffic Accidents With Recent Machine Learning Paradigms and SHAP

Complete bibliographic information and the DOI can be added here following publication.

Contact

For questions concerning the code, please contact the corresponding author through the contact information provided in the associated research article.
