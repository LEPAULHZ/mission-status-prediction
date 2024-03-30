import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from get_matrix import decipher_confusion_matrix
from src.logging import record_metric_to_csv, read_metrics_from_csv, remove_row_from_csv ,clear_csv_file

# Set the option to opt-in to the future behavior
pd.set_option('future.no_silent_downcasting', True)

# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
train_data = pd.read_pickle('../../data/interim/train_data.pkl')
test_data = pd.read_pickle('../../data/interim/test_data.pkl')
balanced_data = pd.read_pickle('../../data/interim/balanced_data.pkl')

# ------------------------------------------------------
# Preprocess
# ------------------------------------------------------
# Handling Rocket Cost Feature Imputation
mean_value = train_data['Rocket Cost'].mean()
median_value = train_data['Rocket Cost'].median()
mode_value = train_data['Rocket Cost'].mode()[0]
imp_strat = median_value

# Data Transformation
# Get columns with only binary values (0 or 1)
binary_columns = balanced_data.columns[balanced_data.isin([0, 1]).all()]

# Get columns with non-binary values
non_binary_columns = balanced_data.columns[~balanced_data.isin([0, 1]).all()]

# Exclude target variable
no_target_variable = balanced_data.columns[balanced_data.columns != 'isMissionSuccess']

binary_columns, len(binary_columns), non_binary_columns, len(non_binary_columns)


# ------------------------------------------------------
# Apply to training set
# ------------------------------------------------------
# Impute missing values in specified columns 
train_data.loc[:,'Rocket Cost'] = train_data['Rocket Cost'].fillna(imp_strat).infer_objects(copy=False)

# Scale the data
train_data[non_binary_columns] = train_data[non_binary_columns].astype(float)
train_data.loc[:, non_binary_columns] = ((train_data[non_binary_columns]-train_data[non_binary_columns].mean())/train_data[non_binary_columns].std())

X_train = train_data.drop('isMissionSuccess', axis=1).values
y_train = train_data['isMissionSuccess'].values
X_train.shape, y_train.shape

# ------------------------------------------------------
# Apply to testing set
# ------------------------------------------------------
# Impute missing values in specified columns 
test_data.loc[:,'Rocket Cost'] = test_data['Rocket Cost'].fillna(imp_strat).infer_objects(copy=False)

# Scale the data only non binary columns
test_data[non_binary_columns] = test_data[non_binary_columns].astype(float)
test_data.loc[:, non_binary_columns] = ((test_data[non_binary_columns]-test_data[non_binary_columns].mean())/test_data[non_binary_columns].std())

X_test = test_data.drop('isMissionSuccess', axis=1).values
y_test = test_data['isMissionSuccess'].values
X_test.shape, y_test.shape

# ------------------------------------------------------
# Model
# ------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

matrix_train = confusion_matrix(y_train_pred, y_train)
matrix_test = confusion_matrix(y_test_pred, y_test)

metrics_train = decipher_confusion_matrix(matrix_train)
metrics_test = decipher_confusion_matrix(matrix_test)

metrics_dict = {
    'Dataset': ['Train', 'Test'],
    'Actual Failure Predicted Failure (TN)': [metrics_train[0], metrics_test[0]],
    'Actual Failure Predicted Success (FP)': [metrics_train[1], metrics_test[1]],
    'Actual Success Predicted Failure (FN)': [metrics_train[2], metrics_test[2]],
    'Actual Success Predicted Success (TP)': [metrics_train[3], metrics_test[3]],
    'Accuracy': [metrics_train[4], metrics_test[4]],
    'Precision': [metrics_train[5], metrics_test[5]],
    'Recall': [metrics_train[6], metrics_test[6]],
    'F1': [metrics_train[7], metrics_test[7]]
}

# Create a DataFrame set 'Dataset' column as index
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.set_index(['Dataset'], inplace=True)

# ------------------------------------------------------
# Logging Metrics
# ------------------------------------------------------
note = ''
observe = ''
action = ''

test_metrics_dict = {
    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'True Negative': metrics_test[0],
    'False Positive': metrics_test[1],
    'False Negative': metrics_test[2],
    'True Positive': metrics_test[3],
    'Accuracy': metrics_test[4],
    'Precision': metrics_test[5],
    'Recall': metrics_test[6],
    'F1': metrics_test[7],
    'Note': note,
    'Observation': observe,
    'Action': action
}

# Define file path
csv_file_path = os.path.join('../../data/metrics', 'test_metric_logreg.csv')

# Record metrics to CSV
#record_metric_to_csv.record_metrics_to_csv(test_metrics_dict, csv_file_path)

# Read metrics from csv
test_metrics_df = pd.DataFrame(read_metrics_from_csv.read_metrics_from_csv(csv_file_path))


# Remove a specfic row from csv 
#remove_row_from_csv.remove_row_from_csv(csv_file_path, #)

# Clear the contents of the csv file
#clear_csv_file.clear_csv_file(csv_file_path)