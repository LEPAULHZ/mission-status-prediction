# Import necessary libraries
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from get_matrix import decipher_confusion_matrix
from src.parameters.get_params import extract_values
from src.logging.record_metrics_to_csv import record_metrics_to_csv
from src.logging.remove_row_from_csv import remove_row_from_csv
from src.logging.clear_csv_file import clear_csv_file
from datetime import datetime
import yaml

# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
# Directory where pickle files are stored
processed_dir = '../../data/processed/'
params_dir = '../../src/parameters/'

# load in master parameter ------------------------------------
# Get a list of existing yaml files
existing_params_files = [filename for filename in os.listdir(params_dir) if filename.endswith('.yaml')]

# Extract the df_number from existing files
existing_params_numbers = [int(filename.split('_')[3].split('.')[0]) for filename in existing_params_files]

# Determine the highest df_number
latest_params_number = max(existing_params_numbers, default=0)

# Construct the filename for the latest YAML file
latest_params_file = f'{params_dir}master_params_df_{latest_params_number}.yaml'

# Load the YAML file directly into a dictionary
with open(latest_params_file, 'r') as yaml_file:
    master_params = yaml.load(yaml_file, Loader=yaml.FullLoader)

# load in data -----------------------------------------------
# Get a list of existing pickle files
existing_df_files = [filename for filename in os.listdir(processed_dir) if filename.endswith('.pkl')]

# Extract the df_number from existing files
existing_df_numbers = [int(filename.split('_')[4].split('.')[0]) for filename in existing_df_files]

# Determine the highest df_number
latest_df_number = max(existing_df_numbers, default=0)

# Load the DataFrame with the latest df_number
X_test = pd.read_pickle(f'{processed_dir}X_test_processed_df_{latest_df_number}.pkl')
X_train = pd.read_pickle(f'{processed_dir}X_train_processed_df_{latest_df_number}.pkl')
y_test = pd.read_pickle(f'{processed_dir}y_test_processed_df_{latest_df_number}.pkl')
y_train = pd.read_pickle(f'{processed_dir}y_train_processed_df_{latest_df_number}.pkl')



# load in specfic master parameter ------------------------------------
# Load the YAML file directly into a dictionary
# with open(f'{params_dir}master_params_df_6.yaml', 'r') as yaml_file:
#     master_params = yaml.load(yaml_file, Loader=yaml.FullLoader)

# # load in specific data -----------------------------------------------
# X_test = pd.read_pickle(f'{processed_dir}X_test_processed_df_6.pkl')
# X_train = pd.read_pickle(f'{processed_dir}X_train_processed_df_6.pkl')
# y_test = pd.read_pickle(f'{processed_dir}y_test_processed_df_6.pkl')
# y_train = pd.read_pickle(f'{processed_dir}y_train_processed_df_6.pkl')


# ------------------------------------------------------
# Model
# ------------------------------------------------------

# Define hyperparameters for Logistic Regression model
logreq_hyperparams = dict(max_iter = 1000)

# Initialize Logistic Regression model
model = LogisticRegression(**logreq_hyperparams)

# Add model & model_hyperparameters to master_params
master_params['model'] = model.__class__.__name__
master_params['model_hyperparameters'] = logreq_hyperparams

# Add time_stamp to master_params
time_stamp = dict(Timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
master_params['time_stamp'] = time_stamp['Timestamp']

# Fit the model
model.fit(X_train, y_train.values.ravel())

# Predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# Calculate AUC scores
auc_score_train = roc_auc_score(y_train, y_train_pred)
auc_score_test = roc_auc_score(y_test, y_test_pred)

# Store AUC metrics in a dictionary
auc_metrics = dict(auc_train = auc_score_train, auc_test = auc_score_test)
master_params['auc_metrics'] = auc_metrics

# Confusion matrices
matrix_train = confusion_matrix(y_train_pred, y_train)
matrix_test = confusion_matrix(y_test_pred, y_test)

# Calculate metrics from confusion matrices
metrics_train = decipher_confusion_matrix(matrix_train)
metrics_test = decipher_confusion_matrix(matrix_test)

# Store performance metrics in a dictionary
metrics_performance = dict(TN_train = metrics_train[0], TN_test = metrics_test[0],
                           FP_train = metrics_train[1], FP_test = metrics_test[1],
                           FN_train = metrics_train[2], FN_test = metrics_test[2],
                           TP_train = metrics_train[3], TP_test = metrics_test[3],
                           accuracy_train = metrics_train[4],accuracy_test = metrics_test[4],
                           precision_train = metrics_train[5], precision_test = metrics_test[5],
                           recall_train = metrics_train[6],recall_test = metrics_test[6],
                           F1_train = metrics_train[7],F1_test = metrics_test[7])

# Update master_params with performance metrics
master_params['metrics_performance'] = metrics_performance

# Extract key-value pairs from master_params
metric_kv_pairs = extract_values(master_params)

# Convert the filtered dictionary to a DataFrame with one row
metrics_df = pd.DataFrame([metric_kv_pairs])

# Define the desired columns to move to the front and back
desired_front_columns = ['time_stamp','dataset_number', 'model', 
                         'accuracy_train', 'accuracy_test', 
                         'precision_train', 'precision_test', 
                         'auc_train','auc_test']

desired_back_columns =  ['TN_train', 'TN_test',
                         'FP_train', 'FP_test', 
                         'FN_train', 'FN_test', 
                         'TP_train', 'TP_test']

# Extract the middle columns
desired_middle_columns = (metrics_df.drop(columns=(desired_front_columns + desired_back_columns))).columns.to_list()

# Reorganize the metrics DataFrame with desired column order
metrics_df = metrics_df[desired_front_columns + desired_middle_columns + desired_back_columns]

# ------------------------------------------------------
# Logging Metrics Copy
# ------------------------------------------------------

# Define file path
csv_file_path = os.path.join('../../data/metrics', 'master_metric_log.csv')
pickle_file_path = os.path.join('../../data/metrics', 'master_metric_log.pkl')

# Read the master metrics from source and make a copy
master_metrics_df = pd.read_pickle(pickle_file_path)
copy_master_metrics_df = master_metrics_df.copy()
copy_master_metrics_df.T

copy_master_metrics_df.rename(columns={'Timestamp': 'time_stamp'}, inplace=True)

# Perform actions on copy version
copy_master_metrics_df = pd.concat([copy_master_metrics_df,metrics_df], ignore_index=True)


# ------------------------------------------------------
# Logging Metrics Actual
# ------------------------------------------------------

# Recording copy master metric to pickle file as master metrics
#copy_master_metrics_df.to_pickle(pickle_file_path)
#copy_master_metrics_df.to_csv(csv_file_path)

# ------------------------------------------------------
# Edit Logging Metrics csv
# ------------------------------------------------------
#record_metrics_to_csv(metrics_df, csv_file_path)

# Remove a specfic row from csv
#remove_row_from_csv.remove_row_from_csv(csv_file_path, #)

# Clear the contents of the csv file
#clear_csv_file.clear_csv_file(csv_file_path)


