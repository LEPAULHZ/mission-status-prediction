import pandas as pd
import numpy as np
from src.features import get_weekend, get_season, get_quarter, get_country_coord, get_matrix

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

# ------------------------------------------------------
# Handling missing values
# ------------------------------------------------------
# Adding NA column {isna:1}
df.loc[:,'Rocket Cost_isna'] = np.where(df['Rocket Cost'].isna(), 1, 0)
# Convert everything to strings and delete commas
df.loc[:,'Rocket Cost'] = df['Rocket Cost'].astype(str).str.replace(',', '').astype(float)

# ------------------------------------------------------
# Handling Date Feature
# ------------------------------------------------------
df.loc[:,'isWeekend'] = get_weekend.get_weekend(df['Date'])
df['isWeekend'].unique()

# ------------------------------------------------------
# Handling Month & Day Feature
# ------------------------------------------------------
df.loc[:,'Season'] = get_season.get_season(df['Date'])
df['Season'].unique()

df.loc[:, 'Quarter'] = get_quarter.get_quarter(df['Date'])
df['Quarter'].unique()

one_hot_datetime = pd.get_dummies(df[['Season','Quarter']]).astype(int)
df = df.join(one_hot_datetime)

# ------------------------------------------------------
# Handling Status Rocket Cost & Ownership Binary Feature
# ------------------------------------------------------
status_rocket_dict = {'Status Rocket': {'StatusActive': 1, 'StatusRetired': 0}}

for feature, bin_map in status_rocket_dict.items():
    df.loc[:, 'isActive'] = df[feature].map(bin_map)
    
ownership_dict = {'Ownership': {'S': 1, 'P': 0}}

for feature, bin_map in ownership_dict.items():
    df.loc[:, 'isStateRun'] = df[feature].map(bin_map)
  
# ------------------------------------------------------  
# Handling Company Origin & Launch Country Feature
# ------------------------------------------------------
company_countries = df['Company Origin'].unique()
launch_countries = df['Launch Country'].unique()

# Concat two countries features together and create unique array
countries = np.unique(np.concatenate((company_countries, launch_countries)))

# Verifying by finding the intesecting between two arrays
intersecting_countries = np.intersect1d(company_countries, launch_countries)
if len(countries) == len(company_countries)+len(launch_countries)-len(intersecting_countries):
    print(f'Array countries succesfully verified. There are {len(countries)} unique countries.')

# Create an empty dict then append the coordinate from each country into the dict
coord_dict = {}
for country in countries:
    coordinates = get_country_coord.get_country_coord(country)
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

column_names = ['Hour', 'Day', 'Year']
time_units = [hour_seconds, day_seconds, year_seconds]  

for column_name, time_unit in zip(column_names, time_units):
    df[column_name + 'Sine'] = np.sin(df['Unix Time'] * 2 * np.pi / time_unit)
    df[column_name + 'Cosine'] = np.cos(df['Unix Time'] * 2 * np.pi / time_unit)
    
# ------------------------------------------------------
# Handling Company Categorical Feature
# ------------------------------------------------------
one_hot_company = pd.get_dummies(df[['Company']]).astype(int)
df = df.join(one_hot_company)

# ------------------------------------------------------
# Handling Location Text Feature
# ------------------------------------------------------
split_locations = df['Location'].str.split(',', expand=True)

for index in split_locations:
    split_locations[index].unique()

# ------------------------------------------------------
# Handling Status Mission Target Feature
# ------------------------------------------------------

df.loc[:, 'isMissionSuccess'] = (df['Status Mission'] == 'Success').astype(int)
df['isMissionSuccess'].unique()

# ======================================================
# New DataFrame
# ======================================================
# Checking original df
df.columns, len(df.columns)

company_col = df.filter(like='Company_')
df_new = df[['Rocket Cost','Rocket Cost_isna', 'isActive', 'isStateRun',
             'Year', 'Month', 'Day', 'Unix Time', 'isWeekend',
             'Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter', 
             'Quarter_Q1', 'Quarter_Q2', 'Quarter_Q3', 'Quarter_Q4',
             'HourSine', 'HourCosine', 'DaySine', 'DayCosine', 'YearSine', 'YearCosine',
             'Company Origin Lat', 'Company Origin Long', 'Launch Country Lat', 'Launch Country Long', 'isMissionSuccess']]
df_new = pd.concat([df_new, company_col], axis=1)

# Checking new df
len(df_new.columns)