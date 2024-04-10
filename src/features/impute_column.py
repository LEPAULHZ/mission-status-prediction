import pandas as pd
from sklearn.impute import SimpleImputer

def impute_data(df_column, imputation_strategy):
    # Define initial values for constant imputation
    fill_value_0 = 0
    fill_value_1 = 1

    # Create SimpleImputer instance with current strategy
    if imputation_strategy.startswith('constant'):
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value_0 if imputation_strategy == 'constant_0' else fill_value_1)
    else:
        imputer = SimpleImputer(strategy=imputation_strategy)

    # Fit and transform the data for the specified column
    imputed_column = imputer.fit_transform(df_column)
    # Get the imputed value after transformation
    imputed_value = imputer.statistics_[0]
    
    # Create dictionary to store the imputed values for each strategy
    imputed_values = dict()
    # Record the imputed value for the current strategy
    imputed_values[f'{df_column.columns[0]}_{imputation_strategy}'] = imputed_value
        
    # Fill in missing values for strategies that were not used
    for strategy in ['mean', 'median', 'most_frequent', 'constant_0', 'constant_1']:
        if f'{df_column.columns[0]}_{strategy}' not in imputed_values:
            imputed_values[f'{df_column.columns[0]}_{strategy}'] = 'NA'
             
    # return dictionary of imputed values and the imputed column
    return imputed_values, imputed_column

if __name__ == "__main__":
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from impute_column import impute_data
        
    # Set the option to opt-in to the future behavior
    pd.set_option('future.no_silent_downcasting', True)
        
    train_data = pd.read_pickle('../../data/interim/train_data.pkl')
        
    # List of imputation strategies to test
    # 'mean', 'median', 'most_frequent', 'constant_0', 'constant_1'
    imputation_strategies = 'mean'
        
    df_rocket_cost = pd.DataFrame(train_data[['Rocket Cost']])
    #
    #rocket_cost_dict,     
    rocket_cost_dict, df_rocket_cost_new = impute_data(df_rocket_cost, imputation_strategies)

    type(df_rocket_cost_new)

    train_data.loc[:,'Rocket Cost'] = df_rocket_cost_new