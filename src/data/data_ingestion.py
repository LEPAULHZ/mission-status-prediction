import pandas as pd
import numpy as np
import os
from src.features.get_country_coord import get_country_coord
from src.features.month_operations import get_quarter, get_season
from src.features.get_weekend import get_weekend

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

# Summary statistics for numerical features
numerical_features = df.select_dtypes(include=[np.number])
numerical_summary_stats = numerical_features.describe().T

# Summary statistics for categorical features
categorical_features = df.select_dtypes(include=[object])
categorical_summary_stats = categorical_features.describe().T
categorical_features.columns

# ------------------------------------------------------
# Handling Status Mission Target Feature
# ------------------------------------------------------

df.loc[:, 'isMissionSuccess'] = (df['Status Mission'] == 'Success').astype(int)
df['isMissionSuccess'].unique()

# ------------------------------------------------------
# Handling Launch Country & Company Origin Feature
# ------------------------------------------------------

# Check countries before
company_countries = df['Company Origin'].unique()
launch_countries = df['Launch Country'].unique()

# Rename mispelled country 'Isreal' to 'Israel'
df.loc[:, 'Company Origin'] = df['Company Origin'].str.replace('Isreal', 'Israel')
# Recategorize 'Arme de l'Air', french air force, to country 'France'
df.loc[:, 'Company Origin'] = df['Company Origin'].str.replace("Arme de l'Air", "France")
# Recategorize 'Sea Launch', multinational company, to country 'USA'
df.loc[:, 'Launch Country'] = df['Launch Country'].str.replace('Sea Launch', 'USA')

# Filter DataFrame for rows where 'Company Origin' is 'Multi'
multi_origin_df = df[df['Company Origin'] == 'Multi']

# Get unique companies in 'Multi' origin
unique_companies_multi_origin = multi_origin_df['Company'].unique()

# Create a dictionary to store counts of each company in 'Multi' origin
company_counts_multi_origin = dict()

# Count occurrences of each company in 'Multi' origin
for company in unique_companies_multi_origin:
    company_filter = multi_origin_df['Company'] == company
    company_counts_multi_origin[company] = company_filter.sum()

# Define the condition to identify rows where 'Company Origin' is 'Multi' and 'Company' is either 'Land Launch' or 'Sea Launch'
company_condition1 = (df['Company Origin'] == 'Multi') & (df['Company'].isin(['Land Launch', 'Sea Launch']))
company_condition1.sum()
# Replace 'Multi' with 'USA' in the 'Company Origin' column where the condition is True
df.loc[company_condition1, 'Company Origin'] = df['Company Origin'].str.replace('Multi', 'USA')
    
# Define the condition to identify rows where 'Company Origin' is 'Multi' and 'Company' is either 'Arianespace', 'ESA', 'CECLES'
company_condition2 = (df['Company Origin'] == 'Multi') & (df['Company'].isin(['Arianespace', 'ESA', 'CECLES']))
company_condition2.sum()
# Replace 'Multi' with 'USA' in the 'Company Origin' column where the condition is True
df.loc[company_condition2, 'Company Origin'] = df['Company Origin'].str.replace('Multi', 'USA')


# Check countries after
company_countries = df['Company Origin'].unique()
launch_countries = df['Launch Country'].unique()

# Concat two countries features together and create unique array
countries = np.unique(np.concatenate((company_countries, launch_countries)))

# Create an empty dict then append the coordinate from each country into the dict
coord_dict = {}
for country in countries:
    coordinates = get_country_coord(country)
    coord_dict[country] = [coordinates[0], coordinates[1]]

# Create a DataFrame by mapping the country features using the coord_dict
# Assign the resulting list of coordinates to two new columns for Lat. and Long.
df[['Company Origin Lat','Company Origin Long']] = pd.DataFrame(df['Company Origin'].map(coord_dict).to_list(), index=df.index)
df[['Launch Country Lat','Launch Country Long']] = pd.DataFrame(df['Launch Country'].map(coord_dict).to_list(), index=df.index)

# ------------------------------------------------------
# Handling DateTime Feature
# ------------------------------------------------------

# Convert object to Timestamp object with UTC Offset %z
date_time_object = pd.to_datetime(df['DateTime'], format="%Y-%m-%d %H:%M:%S%z")

# Convert Timestamp object to Unix timestamp (seconds since Unix epoch)
df.loc[:,'Unix Time'] = (date_time_object - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()

# ------------------------------------------------------
# Handling Cyclical DateTime Feature
# ------------------------------------------------------

hour_seconds = 60*60
day_seconds = hour_seconds*24
year_seconds = day_seconds*365.25

datetime_column_names = ['Hour', 'Day', 'Year']
time_units = [hour_seconds, day_seconds, year_seconds]  

for column_name, time_unit in zip(datetime_column_names, time_units):
    df[column_name + 'Sine'] = np.sin(df['Unix Time'] * 2 * np.pi / time_unit)
    df[column_name + 'Cosine'] = np.cos(df['Unix Time'] * 2 * np.pi / time_unit)

# ------------------------------------------------------
# Handling Month & Day Feature
# ------------------------------------------------------

df.loc[:,'Season'] = get_season(df['Date'])
df['Season'].unique()

df.loc[:, 'Quarter'] = get_quarter(df['Date'])
df['Quarter'].unique()

# ------------------------------------------------------
# Handling Date Feature
# ------------------------------------------------------

df.loc[:,'isWeekend'] = get_weekend(df['Date'])
df['isWeekend'].unique()

# ------------------------------------------------------
# Handling Company Categorical Feature
# ------------------------------------------------------

df['Company'].unique(), df['Company'].nunique()

# ------------------------------------------------------
# Handling missing values
# ------------------------------------------------------

df.loc[:,'Rocket Cost_isna'] = df['Rocket Cost'].isna()
df['Rocket Cost_isna'].unique()

# Convert everything to strings and delete commas (there is one that have comma)
df.loc[:,'Rocket Cost'] = df['Rocket Cost'].astype(str).str.replace(',', '').astype(float)
df['Rocket Cost'].isna().sum()

# ------------------------------------------------------
# New DataFrame 
# ------------------------------------------------------
df.columns
drop_columns = ['Company','Location', 'Detail',
                'Status Mission', 'Launch Country', 'Company Origin', 'Ownership',
                'DateTime', 'Date', 'Time',
                'Launch Country Lat', 'Launch Country Long',
                'HourSine', 'HourCosine', 'DaySine', 'DayCosine', 'YearCosine',
                'Season', 'Quarter', 'Rocket Cost_isna']


df_new = df.drop(columns=drop_columns)
df_new.columns, len(df_new.columns)

# Verifying number of columns in df new
column_number_comparison = pd.DataFrame({'df_old_col': [len(df.columns)],
                                         'df_new_col': [len(df_new.columns)],
                                         'col_drop': [len(drop_columns)]})


# ------------------------------------------------------
# Save Data to Directory
# ------------------------------------------------------

# Directory where pickle files are stored
interim_dir = '../../data/interim/'

# Get a list of existing pickle files
existing_files = [filename for filename in os.listdir(interim_dir) if filename.endswith('.pkl')]

# Extract the df_number from existing files
existing_df_numbers = [int(filename.split('_')[1].split('.')[0]) for filename in existing_files]

# Determine the next available df_number
next_df_number = max(existing_df_numbers, default=0) + 1

# Use the next available df_number
#---------------------------------------------------------------
#df_new.to_pickle(f'{interim_dir}dataframe_{next_df_number}.pkl')
#---------------------------------------------------------------
