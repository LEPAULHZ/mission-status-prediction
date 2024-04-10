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

