import pandas as pd
import numpy as np
import os
from src.features import get_weekend, get_season, get_quarter, get_country_coord

df = pd.read_csv('../../data/raw/global_space_launches.csv')
df.columns
# Columns cleanliness and readability
rename_dict = {'Company Name': 'Company',' Rocket': 'Rocket Cost',
               'Country of Launch': 'Launch Country', 
               'Companys Country of Origin': 'Company Origin', 
               'Private or State Run': 'Ownership'}

df.rename(columns=rename_dict, inplace=True)

# Checking duplicates
df.drop_duplicates()

# Checking dtype and nulls
df.info()

# ------------------------------------------------------
# Handling Status Mission Target Feature
# ------------------------------------------------------

df.loc[:, 'isMissionSuccess'] = (df['Status Mission'] == 'Success').astype(int)
df['isMissionSuccess'].unique()

# ======================================================
# New DataFrame 
# ======================================================
# Summary statistics for numerical features
numerical_features = df.select_dtypes(include=[np.number])
numerical_summary_stats = numerical_features.describe().T

# Summary statistics for categorical features
categorical_features = df.select_dtypes(include=[object])
categorical_summary_stats = categorical_features.describe().T
categorical_features.columns

drop_columns = ['Company', 'Location', 'Detail', 'Status Rocket', 'Rocket Cost',
                'Status Mission', 'Launch Country', 'Company Origin', 'Ownership',
                'DateTime', 'Date', 'Time','Day', 'Month']

df_new = df.drop(columns=drop_columns)

# Verifying number of columns in df new
column_number_comparison = pd.DataFrame({'df_old_col': [len(df.columns)],
                                         'df_new_col': [len(df_new.columns)],
                                         'col_drop': [len(drop_columns)]})


# ======================================================
# Save Data to Directory
# ======================================================
# Directory where pickle files are stored
interim_dir = '../../data/interim/'

# Get a list of existing pickle files
existing_files = [filename for filename in os.listdir(interim_dir) if filename.endswith('.pkl')]

# Extract the df_number from existing files
existing_df_numbers = [int(filename.split('_')[1].split('.')[0]) for filename in existing_files]

# Determine the next available df_number
next_df_number = max(existing_df_numbers, default=0) + 1

# Use the next available df_number
df_new.to_pickle(f'{interim_dir}dataframe_{next_df_number}.pkl')
