# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score
from src.models.get_matrix import decipher_confusion_matrix
from src.parameters.get_params import extract_values
from datetime import datetime
import src.utilities.data_management as manage
from src.visualization.plot_setting import setup_plotting, save_plot
import src.visualization.visualization_utilities as visutil
from sklearn.metrics import mean_absolute_error, mean_squared_error 


# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
result_df = pd.DataFrame()
load_params_dir = '../../src/parameters/'
load_file_extension_params = '.yaml'
_, max_num = manage.get_file_list_and_max_number(load_params_dir, load_file_extension_params)

for file_number in range(1, max_num+1):
    load_base_filename_params = 'master_params_df'
    load_file_number_params = file_number
    master_params = manage.load_from_file(load_params_dir, load_base_filename_params, load_file_extension_params, load_file_number_params)


    load_processed_dir = '../../data/processed/'
    load_base_filenames_processed = ['X_test_processed_df', 
                                    'X_train_processed_df', 
                                    'y_test_processed_df', 
                                    'y_train_processed_df']
    load_file_extension_processed = '.pkl'
    load_file_number_processed = file_number
    X_test = manage.load_from_file(load_processed_dir, load_base_filenames_processed[0], load_file_extension_processed, load_file_number_processed)
    X_train = manage.load_from_file(load_processed_dir, load_base_filenames_processed[1], load_file_extension_processed, load_file_number_processed)
    y_test = manage.load_from_file(load_processed_dir, load_base_filenames_processed[2], load_file_extension_processed, load_file_number_processed)
    y_train = manage.load_from_file(load_processed_dir, load_base_filenames_processed[3], load_file_extension_processed, load_file_number_processed)
    
    # Ensure y_train is a 1-dimensional array
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Convert X_train and X_test to dense format explicitly
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()    
    
    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------
    
    # Define the hyperparameter grids for each model
    logreq_hyperparams = dict(max_iter = 1000)
    svm_hyperparams = dict(kernel = 'linear') 
    rf_hyperparams = dict(random_state = 42)

    # Define models
    models = [(LogisticRegression(), logreq_hyperparams),
              (SVC(), svm_hyperparams),
              (RandomForestClassifier(), rf_hyperparams)]
    
    for model, model_hyperparams in models:
        # Initialize model with hyperparameters
        model.set_params(**model_hyperparams)
        
        # Add model & model_hyperparameters to master_params
        master_params['model'] = model.__class__.__name__
        master_params['model_hyperparams'] = model_hyperparams
        
        model.fit(X_train, y_train)

        # Add time_stamp to master_params
        time_stamp = dict(Timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        master_params['time_stamp'] = time_stamp['Timestamp']

        # Predictions
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # Calculate MSE & store the dictionary
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        mse_metrics = dict(mse_train = mse_train, mse_test = mse_test)
        master_params['mse_metrics'] = mse_metrics
        
        # Calculate MAE & store the dictionary
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mae_metrics = dict(mae_train = mae_train, mae_test = mae_test)
        master_params['mae_metrics'] = mae_metrics
        
        # Calculate AUC scores & store the dictionary
        auc_score_train = roc_auc_score(y_train, y_train_pred)
        auc_score_test = roc_auc_score(y_test, y_test_pred)
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
        
        # Debugging information for the final dataframe
        result_df = pd.concat([result_df, metrics_df], ignore_index=True)
        print(f'Data Frame: {file_number} Analysis Completed for {model.__class__.__name__}')

result_df.columns
len(result_df.columns)

# # Get the coefficients and intercept
# coefficients = model.coef_[0]
# intercept = model.intercept_[0]
# features = X_train.columns

# # Create a DataFrame to store coefficients and intercept
# coefficients_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients, 'Intercept': intercept})

# ------------------------------------------------------
# Logging Metrics Copy
# ------------------------------------------------------

# Define file paths
csv_metrics_file_path = os.path.join('../../data/metrics', 'master_metric_log.csv')
pickle_metrics_file_path = os.path.join('../../data/metrics', 'master_metric_log.pkl')
# old_pickle_metrics_file_path = os.path.join('../../data/metrics', '6.0-master_metric_log.pkl')

# Read the master metrics from source and make a copy
master_metrics_df = pd.read_pickle(pickle_metrics_file_path)
copied_master_metrics_df = master_metrics_df.copy()


## Perform actions on copied version
# copied_master_metrics_df = pd.concat([copied_master_metrics_df, result_df], ignore_index=True)
    
setup_plotting()
visutil.plot_model(copied_master_metrics_df)


# ------------------------------------------------------
# Logging Metrics Actual
# ------------------------------------------------------

# Recording copy master metric to pickle file as master metrics
# copied_master_metrics_df.to_pickle(pickle_metrics_file_path)
# copied_master_metrics_df.to_csv(csv_metrics_file_path)


