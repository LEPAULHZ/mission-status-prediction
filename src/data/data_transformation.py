# Import necessary libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.features.get_imputation import calculate_imputation
import yaml

# Set the option to opt-in to the future behavior
pd.set_option('future.no_silent_downcasting', True)

# ------------------------------------------------------
# Load Data 
# ------------------------------------------------------

# Directory where pickle files are stored
interim_dir = '../../data/interim/'

# Get a list of existing pickle files
existing_files = [filename for filename in os.listdir(interim_dir) if filename.endswith('.pkl')]

# Extract the df_number from existing files
existing_df_numbers = [int(filename.split('_')[1].split('.')[0]) for filename in existing_files]

# Determine the highest df_number
latest_df_number = max(existing_df_numbers, default=0)

# Load the DataFrame with the latest df_number
df_new = pd.read_pickle(f'{interim_dir}dataframe_{latest_df_number}.pkl')

# ------------------------------------------------------
# Preprocess
# ------------------------------------------------------
drop_columns = ['Company','Location', 'Detail',
                'Status Mission', 'Launch Country', 'Company Origin', 'Ownership',
                'DateTime', 'Date', 'Time',
                'Launch Country Lat', 'Launch Country Long',
                'HourSine', 'HourCosine', 'DaySine', 'DayCosine', 'YearCosine',
                'Season', 'Quarter', 'Rocket Cost_isna',
                'Pad', 'Center', 'State', 'Country', 'Rocket', 'Mission']

df_new_columns = df_new.drop(columns=drop_columns)
df_new_columns.columns, len(df_new_columns.columns)

X = df_new_columns.drop(columns=['isMissionSuccess'])
y = df_new_columns['isMissionSuccess']

# Creating a master parameter dictionary
master_params = dict()

# Split data to training and testing
train_test_split_params = dict(test_size = 0.05, random_state = 42)
master_params['train_test_split'] = train_test_split_params
X_train, X_test, y_train, y_test = train_test_split(X, y, **train_test_split_params)

random_oversampler_params = dict(sampling_strategy = 'minority', random_state = 42)
master_params['random_oversampler'] = random_oversampler_params
# Initialize RandomOverSampler object
random_oversampler = RandomOverSampler(**random_oversampler_params)

# Balance the training and testind data
X_train_resampled, y_train_resampled = random_oversampler.fit_resample(X_train, y_train)
X_test_resampled, y_test_resampled = random_oversampler.fit_resample(X_test, y_test)

X_train_resampled.shape, y_train_resampled.shape
X_test_resampled.shape, y_test_resampled.shape

# ------------------------------------------------------
# Define transformers 
# ------------------------------------------------------

columns_total = df_new_columns.columns.tolist()

# Define columns for imputation, scaling, binary encoding, and one-hot encoding
columns_imputation_mean = []
columns_imputation_median = ['Rocket Cost']
columns_imputation_mode = []
columns_imputation_constant1 = []
columns_imputation_constant0 = []
columns_scaling = ['Year', 'Month', 'Day', 'Company Origin Lat', 'Company Origin Long', 'Unix Time', 'YearSine']
columns_binary = ['Status Rocket', 'isWeekend']
columns_ohe = []

# Define hyperparameters for OneHotEncoder
ohe_hyperparams = dict(handle_unknown = 'ignore')
ohe = OneHotEncoder(**ohe_hyperparams)
master_params['ohe_hyperparams'] = ohe_hyperparams

# Calculate imputation hyperparameters for mean
impute_hyperparams_mean = dict()
for col in columns_imputation_mean:
    impute_hyperparams_mean[col] = {'strategy': 'constant', 'fill_value': calculate_imputation(col, X_train_resampled[col], 'mean')}
    
# Calculate imputation hyperparameters for median
impute_hyperparams_median = dict()
for col in columns_imputation_median:
    impute_hyperparams_median[col] = {'strategy': 'constant', 'fill_value': calculate_imputation(col, X_train_resampled[col], 'median')}
    
# Calculate imputation hyperparameters for mode
impute_hyperparams_mode = dict()
for col in columns_imputation_mode:
    impute_hyperparams_mode[col] = {'strategy': 'constant', 'fill_value': calculate_imputation(col, X_train_resampled[col], 'mode')}
    
# Calculate imputation hyperparameters for constant 1
impute_hyperparams_constant1 = dict()
for col in columns_imputation_constant1:
    impute_hyperparams_constant1[col] = {'strategy': 'constant', 'fill_value': 1}

# Calculate imputation hyperparameters for constant 0
impute_hyperparams_constant0 = dict()
for col in columns_imputation_constant0:
    impute_hyperparams_constant0[col] = {'strategy': 'constant', 'fill_value': 0}

# Aggregate imputation hyperparameters into a single dictionary
imputer_hyperparams = dict(mean = impute_hyperparams_mean,
                           median = impute_hyperparams_median,
                           mode = impute_hyperparams_mode,
                           constant1 = impute_hyperparams_constant1,
                           constant0 = impute_hyperparams_constant0)

# Log imputation hyperparameters for monitoring/logging
imputer_values_logging = dict()
for strategy, feature_dict in imputer_hyperparams.items():
    if strategy == 'constant0' or strategy == 'constant1':
        for feature, params in feature_dict.items():
            imputer_values_logging[f'{feature} {strategy}'] = True
    else:     
        for feature, params in feature_dict.items():
            if feature:
                imputer_values_logging[f'{feature} {strategy}'] = float(params['fill_value'])

# Add imputation_hyperparams to master_params
master_params['imputation_hyperparams'] = imputer_values_logging            
   
            
# Create SimpleImputer instances and pipelines for each column and strategy
imputer_pipelines = []
columns_imputer = []
for strategy, feature_dict in imputer_hyperparams.items():
    for feature, params in feature_dict.items():
        # Create SimpleImputer instance with specified parameters
        imputer = SimpleImputer(strategy=params['strategy'], fill_value=params['fill_value'])
        # Create pipeline with SimpleImputer
        pipeline = make_pipeline(imputer)
        # Append SimpleImputer instance and pipeline to lists
        imputer_pipelines.append(pipeline)
        columns_imputer.append(feature)

# Initialize pipelines for binary encoding and scaling
binary = OrdinalEncoder()
scaler = StandardScaler()

# Create pipelines for binary encoding and scaling
binary_pipeline = make_pipeline(binary)
scaler_pipeline = make_pipeline(scaler)
ohe_pipeline = make_pipeline(ohe)

# Create columntransformer
ct = make_column_transformer((scaler_pipeline, columns_scaling),
                             (binary_pipeline, columns_binary),
                             (imputer_pipelines[0], [columns_imputer[0]]),
                             (ohe_pipeline, columns_ohe)
                             )

transformer_columns = dict()
# Iterate over transformers in the ColumnTransformer
for i, (name, transformer, columns) in enumerate(ct.transformers):
    # If transformer is a pipeline, get the last step (which is the actual transformer)
    if hasattr(transformer, 'steps'):
        transformer = transformer.steps[-1][1]
    
    # Get the name of the transformer class
    transformer_name = transformer.__class__.__name__
    
    # Add transformer name and associated columns to the dictionary
    if transformer_name not in transformer_columns:
        transformer_columns[transformer_name] = columns

    else:
        # If transformer name already exists in the dictionary, extend the columns list
        transformer_columns[transformer_name].extend(columns)

    
# Add column_transformations to master_params
master_params['column_transformations'] = transformer_columns

# ------------------------------------------------------
# Apply to training and testing set
# ------------------------------------------------------

X_train_processed = ct.fit_transform(X_train_resampled)
X_test_processed = ct.transform(X_test_resampled)
X_train_processed.shape, X_test_processed.shape

# ------------------------------------------------------
# Saving data frames, transformers, parameters 
# ------------------------------------------------------

# Define directories
processed_dir = '../../data/processed/'
params_dir = '../../src/parameters/'

# Get a list of existing pickle files
existing_files_processed = [filename for filename in os.listdir(processed_dir) if filename.endswith('.pkl')]
# Extract the df_number from existing files
existing_processed_numbers = [int(filename.split('_')[4].split('.')[0]) for filename in existing_files_processed]
# Determine the highest df_number
latest_processed_number = max(existing_processed_numbers, default=0) + 1

# Get a list of existing pickle files
existing_files_params = [filename for filename in os.listdir(params_dir) if filename.endswith('.yaml')]
# Extract the df_number from existing files
existing_params_numbers = [int(filename.split('_')[3].split('.')[0]) for filename in existing_files_params]
# Determine the highest df_number
latest_params_number = max(existing_params_numbers, default=0) + 1

# Convert arrays to DataFrame
X_train_processed_df = pd.DataFrame(X_train_processed, columns=X_train_resampled.columns)
y_train_resampled_df = pd.DataFrame(y_train_resampled)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=X_test_resampled.columns)
y_test_resampled_df = pd.DataFrame(y_test_resampled)

# Save as pickle files latest
# X_train_file = f'{processed_dir}X_train_processed_df_{latest_processed_number}.pkl'
# y_train_file = f'{processed_dir}y_train_processed_df_{latest_processed_number}.pkl'
# X_test_file = f'{processed_dir}X_test_processed_df_{latest_processed_number}.pkl'
# y_test_file = f'{processed_dir}y_test_processed_df_{latest_processed_number}.pkl'

# X_train_processed_df.to_pickle(X_train_file)
# y_train_resampled_df.to_pickle(y_train_file)
# X_test_processed_df.to_pickle(X_test_file)
# y_test_resampled_df.to_pickle(y_test_file)

# Save master_params along with dataset number and ColumnTransformer to a YAML file
master_params_file = f'{params_dir}master_params_df_{latest_processed_number}.yaml'

#----------------------------------------------------------------------------------------
# with open(master_params_file, 'w') as f:
#    yaml.dump({'dataset_number': latest_df_number, 'master_params': master_params}, f)
#----------------------------------------------------------------------------------------
