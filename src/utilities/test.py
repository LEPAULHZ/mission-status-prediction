import pandas as pd
from sklearn.impute import SimpleImputer

# Set the option to opt-in to the future behavior
pd.set_option('future.no_silent_downcasting', True)

# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
train_data = pd.read_pickle('../../data/interim/train_data.pkl')
test_data = pd.read_pickle('../../data/interim/test_data.pkl')
balanced_data = pd.read_pickle('../../data/interim/balanced_data.pkl')

# Define initial values for constant imputation
constant_fill_value_0 = 0
constant_fill_value_1 = 1

# List of imputation strategies to test
# 'mean', 'median', 'most_frequent', 'constant_0', 'constant_1'
rocket_cost_imputation_strategies = ['mean']

# Create dictionary to store the imputed values for each strategy
imputed_values_dict = {}

# Loop through each imputation strategy
for imp_strat in rocket_cost_imputation_strategies:
    # Create SimpleImputer instance with current strategy
    if imp_strat.startswith('constant'):
        imputer = SimpleImputer(strategy='constant', fill_value=constant_fill_value_0 if imp_strat == 'constant_0' else constant_fill_value_1)
    else:
        imputer = SimpleImputer(strategy=imp_strat)

    # Fit and transform the data for Rocket Cost column
    train_data_imputed_rocket_cost = imputer.fit_transform(train_data[['Rocket Cost']])
    
    # Get the imputed value after transformation
    imputed_value = imputer.statistics_[0]  # Assuming there's only one value for the Rocket Cost column

    # Record the imputed value for the current strategy
    imputed_values_dict[f'imp_strat_{imp_strat}'] = imputed_value

# Fill in missing values for strategies that were not used
for imp_strat in ['mean', 'median', 'most_frequent', 'constant_0', 'constant_1']:
    if f'imp_strat_{imp_strat}' not in imputed_values_dict:
        imputed_values_dict[f'imp_strat_{imp_strat}'] = 'NA'

# Create a DataFrame with a single row from the recorded information for Rocket Cost column
df_rocket_cost_imp_strat_changes = pd.DataFrame(imputed_values_dict, index=[0])





