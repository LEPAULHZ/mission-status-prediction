# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import sys
import os
sys.path.append('.')
from plot_setting import setup_plotting, save_plot
import categorical_plots as catplt
import text_processing as txtpro
import visualization_utilities as visutil

# ------------------------------------------------------
# Import original DF
# ------------------------------------------------------

df_original = pd.read_csv('../../data/raw/global_space_launches.csv')
df_original.columns
# Columns cleanliness and readability
rename_dict = {'Company Name': 'Company',' Rocket': 'Rocket Cost',
               'Country of Launch': 'Launch Country', 
               'Companys Country of Origin': 'Company Origin', 
               'Private or State Run': 'Ownership'}

df_original.rename(columns=rename_dict, inplace=True)

# Checking duplicates
df_original.drop_duplicates()

# Checking dtype and nulls
df_original.info()

# Summary statistics for numerical features
numerical_features = df_original.select_dtypes(include=[np.number])
numerical_summary_stats = numerical_features.describe().T
numerical_features.columns

# Summary statistics for categorical features
categorical_features = df_original.select_dtypes(include=[object])
categorical_summary_stats = categorical_features.describe().T
categorical_features.columns

# Setup plotting preferences
setup_plotting()

# Toggle for saving plots
SAVE_PLOTS = False  

# Target Variable Status Mission -----------------------
# Group all failures into 1
status_mission_binary = df_original['Status Mission'].replace(['Partial Failure', 'Failure', 'Prelaunch Failure'], 'Failure')
status_mission = df_original[['Status Mission']]
# Plot
ax_status_mission = catplt.plot_countplots_y(status_mission)
save_plot(ax_status_mission, ax_status_mission.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature Variable Company -----------------------------
# Categorical with 55 unique values consider grouping
company = df_original[['Company']]
company['Company'].nunique()
# Plot
plt.figure(figsize=(20,20))
ax_company = catplt.plot_countplots_y_ordered(company)
ax_company.set_xscale('log')
save_plot(ax_company, ax_company.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature Variable Location ----------------------------
# NLP with 137 unique values useless countplot string split
location = df_original[['Location']]
location['Location'].nunique()
# Plot
plt.figure(figsize=(20,40))
ax_location = catplt.plot_countplots_y_ordered(location)
ax_location.set_xscale('log')
save_plot(ax_location, ax_location.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature Variable Status Rocket -----------------------
status_rocket = df_original[['Status Rocket']]
# Plot
ax_status_rocket = catplt.plot_countplots_y(status_rocket)
save_plot(ax_status_rocket, ax_status_rocket.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature Variable Rocket Cost -------------------------
# Numerical with NaNs and high side outliers
rocket_cost = df_original['Rocket Cost'].astype(str).str.replace(',', '').astype(float)
rocket_cost = rocket_cost.to_frame(name='Rocket Cost')
# Plot
plt.figure(figsize=(10, 5)) 
ax1_rocket_cost = sns.boxplot(y=rocket_cost['Rocket Cost'], log_scale=True)
ax1_rocket_cost.set_title('Boxplot Distribution of Rocket Cost')
ax1_rocket_cost.set_ylabel('Rocket Cost (USD Millions)')
save_plot(ax1_rocket_cost, ax1_rocket_cost.get_title(), save=SAVE_PLOTS) 
plt.show()
plt.close()

ax2_rocket_cost = sns.histplot(rocket_cost['Rocket Cost'], log_scale=True, kde=True)
ax2_rocket_cost.set_title('Hisplot Launch Number Distribution of Rocket Cost') 
ax2_rocket_cost.set_xlabel('Rocket Cost (USD Millions)') 
ax2_rocket_cost.set_ylabel('Launch Number')
save_plot(ax2_rocket_cost, ax2_rocket_cost.get_title(), save=SAVE_PLOTS) 
plt.show()
plt.close()

# Feature Variable Launch Country ----------------------
# Categorical with 16 unique values
launch_country = df_original[['Launch Country']]
launch_country['Launch Country'].nunique()
# Plot
ax_launch_country = catplt.plot_countplots_y_ordered(launch_country)
ax_launch_country.set_xscale('log')
save_plot(ax_launch_country, ax_launch_country.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close
launch_country['Launch Country'].value_counts()
# Feature Variable Company Origin ----------------------
# Categorical with 17 unique values
company_origin = df_original[['Company Origin']]
company_origin['Company Origin'].nunique()
# Plot
ax_company_origin = catplt.plot_countplots_y_ordered(company_origin)
ax_company_origin.set_xscale('log')
save_plot(ax_company_origin, ax_company_origin.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature Variable Ownership ---------------------------
ownership = df_original[['Ownership']]
# Plot
ax_ownership = catplt.plot_countplots_y(ownership)
save_plot(ax_ownership, ax_ownership.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature Variable Year --------------------------------
# High Cardinality with 64 unique values
year = df_original[['Year']]
year['Year'].nunique()
# Plot
plt.figure(figsize=(20,30))
ax_year = catplt.plot_countplots_y(year)
ax_year.legend_.remove()
save_plot(ax_year, ax_year.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature Variable Month -------------------------------
month = df_original[['Month']]
# Plot
ax_month = catplt.plot_countplots_y(month)
ax_month.legend_.remove()
save_plot(ax_month, ax_month.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature Variable Day ---------------------------------
day = df_original[['Day']]
# Plot
plt.figure(figsize=(20,10))
ax_day = catplt.plot_countplots_y(day)
ax_day.legend_.remove()
save_plot(ax_day, ax_day.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close
len(df_original.columns)
# Feature Variable DateTime ----------------------------
# Time-series with 4319 unique values cannot plot

# Feature Variable Date --------------------------------
# Time-series with 3922 unique values cannot plot

# Feature Variable Time --------------------------------
# Time-series with 1273 unique values cannot plot

# Feature Variable Detail ------------------------------
# NLP with 4278 unique values cannot plot string split


# ------------------------------------------------------
# Import New DF
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

# Checking duplicates
df_new.drop_duplicates()

# Checking dtype and nulls
df_new.info()

# Get new columns 
new_columns = sorted(list(set(df_new.columns.to_list()) - set(df_original.columns.to_list())))


# Feature Company Origin Lat/Long ----------------------
map_company_origin = df_new.groupby(['Company Origin Long', 'Company Origin Lat'])['Status Mission'].count().reset_index()
# Plot
plt.figure(figsize=(20,10))
ax_company_origin_geo = sns.scatterplot(data=map_company_origin,
                                        x='Company Origin Long', 
                                        y='Company Origin Lat',
                                        size='Status Mission',
                                        sizes=(100, 1000),
                                        legend=False)
ax_company_origin_geo.set_xlim(-180, 180)  
ax_company_origin_geo.set_ylim(-90, 90)  
ax_company_origin_geo.set_title('Scatterplot Geographical Distribution of Company Origin')
ax_company_origin_geo.set_xlabel('Longitude') 
ax_company_origin_geo.set_ylabel('Latitude') 
save_plot(ax_company_origin_geo, ax_company_origin_geo.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature Launch Origin Lat/Long -----------------------
map_launch_country = df_new.groupby(['Launch Country Long', 'Launch Country Lat'])['Status Mission'].count().reset_index()
# Plot
plt.figure(figsize=(20,10))
ax_launch_country_geo = sns.scatterplot(data=map_launch_country, 
                                        x='Launch Country Long', 
                                        y='Launch Country Lat',
                                        size='Status Mission',
                                        sizes=(100, 1000), 
                                        legend=False)
ax_launch_country_geo.set_xlim(-180, 180)  
ax_launch_country_geo.set_ylim(-90, 90)   
ax_launch_country_geo.set_title('Scatterplot Geographical Distribution of Launch Country')
ax_launch_country_geo.set_xlabel('Longitude') 
ax_launch_country_geo.set_ylabel('Latitude')
save_plot(ax_launch_country_geo, ax_launch_country_geo.get_title(), save=SAVE_PLOTS) 
plt.show()
plt.close()

# Feature HourCosine/Sine ------------------------------
hour_cosine = df_new['HourCosine']
hour_sine = df_new['HourSine']
# Plot
plt.figure(figsize=(20,20))
ax_hour = sns.scatterplot(x=hour_sine, y=hour_cosine)
ax_hour.set_title('Scatterplot Cyclical of Hour in Seconds')
ax_hour.set_xlabel('Sine of Hour in Seconds') 
ax_hour.set_ylabel('Cosine of Hour in Seconds')
save_plot(ax_hour, ax_hour.get_title(), save=SAVE_PLOTS) 
plt.show()
plt.close()

# Feature DayCosine/Sine -------------------------------
day_cosine = df_new['DayCosine']
day_sine = df_new['DaySine']
# Plot
plt.figure(figsize=(20,20))
ax_day = sns.scatterplot(x=day_sine, y=day_cosine)
ax_day.set_title('Scatterplot Cyclical of Day in Seconds')
ax_day.set_xlabel('Sine of Day in Seconds') 
ax_day.set_ylabel('Cosine of Day in Seconds')
save_plot(ax_day, ax_day.get_title(), save=SAVE_PLOTS)  
plt.show()
plt.close()

# Feature YearCosine/Sine ------------------------------
year_cosine = df_new['YearCosine']
year_sine = df_new['YearSine']
# Plot
plt.figure(figsize=(20,20))
ax_year = sns.scatterplot(x=year_sine, y=year_cosine)
ax_year.set_title('Scatterplot Cyclical of Year in Seconds')
ax_year.set_xlabel('Sine of Year in Seconds') 
ax_year.set_ylabel('Cosine of Year in Seconds')
save_plot(ax_year, ax_year.get_title(), save=SAVE_PLOTS) 
plt.show()
plt.close()

# Feature Unix Time ------------------------------------
# Time-series better to plot with second variable
unix_time = df_new[['Unix Time']]
# Plot
ax_unix_time = sns.histplot(x=unix_time['Unix Time'], kde=True)
ax_unix_time.set_title('Hisplot Launch Number Distribution of Unix Time')
ax_unix_time.set_xlabel('Unix Time (s)') 
ax_unix_time.set_ylabel('Launch Number')
save_plot(ax_unix_time, ax_unix_time.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature Rocket Cost_isna -----------------------------
rocket_cost_isna = df_new[['Rocket Cost_isna']]
# Plot
ax_rocket_cost_isna = catplt.plot_countplots_y(rocket_cost_isna)
ax_rocket_cost_isna.legend_.remove()
save_plot(ax_rocket_cost_isna, ax_rocket_cost_isna.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature isMissionSuccess -----------------------------
is_mission_success = df_new[['isMissionSuccess']]
# Plot
ax_is_mission_success = catplt.plot_countplots_y(is_mission_success)
ax_is_mission_success.legend_.remove()
save_plot(ax_is_mission_success, ax_is_mission_success.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature isWeekend ------------------------------------
is_weekend = df_new[['isWeekend']]
# Plot
ax_is_weekend = catplt.plot_countplots_y(is_weekend)
ax_is_weekend.legend_.remove()
save_plot(ax_is_weekend, ax_is_weekend.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature Season ---------------------------------------
season = df_new[['Season']]
# Plot
ax_season = catplt.plot_countplots_y(season)
save_plot(ax_season, ax_season.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature Quarter --------------------------------------
quarter = df_new[['Quarter']]
# Plot
ax_quarter = catplt.plot_countplots_y(quarter)
save_plot(ax_quarter,ax_quarter.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature Center ---------------------------------------
# High Cardinality with 42 unique values
center = df_new[['Center']]
center['Center'].nunique()
# Plot
plt.figure(figsize=(20,10))
ax_center = catplt.plot_countplots_y_ordered(center)
save_plot(ax_center, ax_center.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close

# Feature Pad ------------------------------------------
# NLP with 124 unique values useless plot
pad = df_new[['Pad']]
pad['Pad'].nunique()
# Plot
plt.figure(figsize=(20,40))
ax_pad = catplt.plot_countplots_y_ordered(pad)
save_plot(ax_pad, ax_pad.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature State ----------------------------------------
# Categorical with 12 unique values
state = df_new[['State']]
# Plot
ax_state = catplt.plot_countplots_y_ordered(state)
ax_state.set_xscale('log')
save_plot(ax_state, ax_state.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close
state['State'].value_counts()
# Feature Country --------------------------------------
# Original 'Launch Country' feature represents the new 'Country' so we can exclude this

# Feature Mission --------------------------------------
# NLP with 4257 unique values cannot plot
mission = df_new[['Mission']]
mission['Mission'].nunique()

mission_term_matrix, mission_count_vectorizer = txtpro.compute_term_matrix(mission['Mission'])
num_mission_range = range(1, min(mission_term_matrix.shape)+1)
mission_cumulative_variance = txtpro.compute_cumulative_variance(mission_term_matrix)
# Plot the cumulative explained variance ratio
ax_mission = sns.scatterplot(x=num_mission_range, y=mission_cumulative_variance)
ax_mission.set_title('Scatterplot Variance by Number of SVD Components for Mission')
ax_mission.set_xlabel('Number of Components')
ax_mission.set_ylabel('Cumulative Variance Ratio')
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(100))

# Finding the elbow point
mission_tol = 1e-3
mission_diffs = np.diff(mission_cumulative_variance)
mission_elbow_index = np.argmax(mission_diffs < mission_tol)  
mission_elbow_point = num_mission_range[mission_elbow_index]
# Mark the elbow point on the plot
plt.axvline(x=mission_elbow_point, color='r', linestyle='--', label=f'Elbow Point: {mission_elbow_point}')
plt.legend()
save_plot(ax_mission, ax_mission.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()


# Feature Rocket ---------------------------------------
# NLP with 352 unique values useless countplot
rocket = df_new[['Rocket']]
rocket['Rocket'].nunique()

rocket_term_matrix, rocket_count_vectorizer = txtpro.compute_term_matrix(rocket['Rocket'])
num_rocket_range = range(1, min(rocket_term_matrix.shape)+1)
rocket_cumulative_variance = txtpro.compute_cumulative_variance(rocket_term_matrix)
# Plot the cumulative explained variance ratio
ax_rocket = sns.scatterplot(x=num_rocket_range, y=rocket_cumulative_variance)
ax_rocket.set_title('Scatterplot Variance by Number of SVD Components for Rocket')
ax_rocket.set_xlabel('Number of Components')
ax_rocket.set_ylabel('Cumulative Variance Ratio')
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(10))

# Finding the elbow point
rocket_tol = 1e-2
rocket_diffs = np.diff(rocket_cumulative_variance)
rocket_elbow_index = np.argmax(rocket_diffs < rocket_tol)  
rocket_elbow_point = num_rocket_range[rocket_elbow_index]
# Mark the elbow point on the plot
plt.axvline(x=rocket_elbow_point, color='r', linestyle='--', label=f'Elbow Point: {rocket_elbow_point}')
plt.legend()
save_plot(ax_rocket, ax_rocket.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()


# ------------------------------------------------------
# Plot Feature vs Target
# ------------------------------------------------------

# Feature Company vs Status Mission --------------------
company_success_percent, company_failure_percent = visutil.mission_percent(df_original, 'Company')
ax_company_percent = catplt.plot_bar(company_success_percent, company_failure_percent, 'Company')
save_plot(ax_company_percent, ax_company_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()

# Feature Launch Country vs Status Mission -------------
lc_success_percent, lc_failure_percent = visutil.mission_percent(df_original, 'Launch Country')
ax_launch_country_percent = catplt.plot_bar(lc_success_percent, lc_failure_percent, 'Launch Country')
save_plot(ax_launch_country_percent, ax_launch_country_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()

# Feature Company Origin vs Status Mission -------------
co_success_percent, co_failure_percent = visutil.mission_percent(df_original, 'Company Origin')
ax_company_origin_percent = catplt.plot_bar(co_success_percent, co_failure_percent, 'Company Origin')
save_plot(ax_company_origin_percent, ax_company_origin_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()

# Feature Year vs Status Mission -----------------------
year_success_percent, year_failure_percent = visutil.mission_percent(df_original, 'Year')
year_percent = catplt.plot_bar(year_success_percent, year_failure_percent, 'Year')
plt.axhline(y=np.mean(year_success_percent.values), 
            color='black', 
            linestyle='--', 
            label=f'Success Mean: {round(np.mean(year_success_percent.values),2)}')
plt.axhline(y=np.mean(year_failure_percent.values), 
            color='black', 
            linestyle='--', 
            label=f'Failure Mean: {round(np.mean(year_failure_percent.values), 2)}')
plt.legend()
save_plot(year_percent, year_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()

# Feature Month vs Status Mission ----------------------
month_success_percent, month_failure_percent = visutil.mission_percent(df_original, 'Month')
month_percent = catplt.plot_bar(month_success_percent, month_failure_percent, 'Month')
save_plot(month_percent, month_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()

# Feature Day vs Status Mission ------------------------
day_success_percent, day_failure_percent = visutil.mission_percent(df_original, 'Day')
day_percent = catplt.plot_bar(day_success_percent, day_failure_percent, 'Day')
save_plot(day_percent, day_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()


# Feature Status Rocket vs Status Mission --------------

# Contingency Table
table1 = pd.crosstab(df_original['Status Rocket'], status_mission_binary )
# Plot
sr_mission = sns.heatmap(table1, annot=True, cmap='Blues', fmt='g')
sr_mission.set_title('Heatmap Rocket Status by Status Mission')
sr_mission.set_xlabel('Status Mission')
sr_mission.set_ylabel('Rocket Status')
save_plot(sr_mission, sr_mission.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature Ownership vs Status Mission ------------------
# Contingency Table
table2 = pd.crosstab(df_original['Ownership'], status_mission_binary )
# Plot
ownership_mission = sns.heatmap(table2, annot=True, cmap='Blues', fmt='g')
ownership_mission.set_title('Heatmap Ownership by Status Mission')
ownership_mission.set_xlabel('Status Mission')
ownership_mission.set_ylabel('Ownership')
save_plot(ownership_mission, ownership_mission.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature Rocket Cost vs Status Mission ----------------
rocket_cost_drop = df_original[['Status Mission', 'Rocket Cost']]
rocket_cost_drop.loc[:,'Status Mission'] = rocket_cost_drop['Status Mission'].replace(['Partial Failure', 'Failure', 'Prelaunch Failure'], 'Failure')
rocket_cost_drop = rocket_cost_drop.dropna()
rocket_cost_drop.loc[:,'Rocket Cost'] = rocket_cost_drop['Rocket Cost'].astype(str).str.replace(',', '').astype(float)

ax1_rc_mission = sns.kdeplot(data=rocket_cost_drop, x='Rocket Cost', hue='Status Mission', fill=True, log_scale=True)
ax1_rc_mission.set_title('Kdeplot Density Status Mission Distribution of Rocket Cost ')
ax1_rc_mission.set_xlabel('Rocket Cost (USD Millions)')
ax1_rc_mission.set_ylabel('Density')
save_plot(ax1_rc_mission, ax1_rc_mission.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

ax2_rc_mission = sns.boxplot(x='Rocket Cost',y='Status Mission', data=rocket_cost_drop, log_scale=True)
ax2_rc_mission.set_title('Boxplot Status Mission Distribution of Rocket Cost ')
ax2_rc_mission.set_xlabel('Rocket Cost (USD Millions)')
save_plot(ax2_rc_mission, ax2_rc_mission.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature Center vs Status Mission ---------------------
center_success_percent, center_failure_percent = visutil.mission_percent(df_new, 'Center')
center_percent = catplt.plot_bar(center_success_percent, center_failure_percent, 'Center')
save_plot(center_percent, center_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()

# Feature Quarter vs Status Mission --------------------
quarter_success_percent, quarter_failure_percent = visutil.mission_percent(df_new, 'Quarter')
quarter_percent = catplt.plot_bar(quarter_success_percent, quarter_failure_percent, 'Quarter')
save_plot(quarter_percent, quarter_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()

# Feature Season vs Status Mission ---------------------
season_success_percent, season_failure_percent = visutil.mission_percent(df_new, 'Season')
season_percent = catplt.plot_bar(season_success_percent, season_failure_percent, 'Season')
save_plot(season_percent, season_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()

# Feature isWeekend vs Status Mission ------------------
# Contingency Table
table3 = pd.crosstab(df_new['isWeekend'], status_mission_binary )
# Plot
isweekend_mission = sns.heatmap(table3, annot=True, cmap='Blues', fmt='g')
isweekend_mission.set_title('Heatmap isWeekend by Status Mission')
isweekend_mission.set_xlabel('Status Mission')
isweekend_mission.set_ylabel('isWeekend')
save_plot(isweekend_mission, isweekend_mission.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()

# Feature State vs Status Mission ----------------------
state_success_percent, state_failure_percent = visutil.mission_percent(df_new, 'State')
state_percent = catplt.plot_bar(state_success_percent, state_failure_percent, 'State')
save_plot(state_percent, state_percent.get_title(), save=SAVE_PLOTS)
plt.show()  
plt.close()

# Feature Unix Time vs Status Mission ------------------
ax_unixtime_misssion = sns.histplot(x=df_new['Unix Time'], hue=status_mission_binary, kde=True)
ax_unixtime_misssion.set_title('Histplot Distribution Status Mission of Unix Time')
ax_unixtime_misssion.set_xlabel('Unix Time (s)')
ax_unixtime_misssion.set_ylabel('Launch Number')
save_plot(ax_unixtime_misssion, ax_unixtime_misssion.get_title(), save=SAVE_PLOTS)
plt.show()
plt.close()