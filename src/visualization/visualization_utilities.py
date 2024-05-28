import matplotlib.pyplot as plt
import seaborn as sns

def mission_percent(df, column_name):
    # Count total, success, and failure missions directly
    mission_count = df[column_name].value_counts().sort_index()
    success_counts = df[df['Status Mission'] == 'Success'].groupby(column_name).size() 
    failure_counts = df[df['Status Mission'] != 'Success'].groupby(column_name).size() 

    # Compute percentages
    success_percent = (success_counts / mission_count * 100).reindex(mission_count.index).fillna(0)
    failure_percent = (failure_counts / mission_count * 100).reindex(mission_count.index).fillna(0)
    
    return success_percent, failure_percent

def plot_model(master_metrics):
    for model in master_metrics['model'].unique():
        # Filter for the current model
        filtered_metrics = master_metrics[master_metrics['model'] == model]
        
        # Group by 'dataset_number' and find the index of the maximum 'auc_test' value for each group
        idx_max_auc = filtered_metrics.groupby('dataset_number')['auc_test'].idxmax()
        
        # Use these indices to filter the DataFrame
        filtered_max_auc = filtered_metrics.loc[idx_max_auc]
        
        # Reshaping the df from wide to long format
        filtered_melted = filtered_max_auc.melt(id_vars=['dataset_number'], value_vars=['auc_train', 'auc_test'], var_name='AUC_Type', value_name='AUC_Value')
        filtered_melted['AUC_Value'] = filtered_melted['AUC_Value'].astype(float).round(4)
        
        # Plot the results
        ax = sns.barplot(data=filtered_melted, x='dataset_number', y='AUC_Value', hue='AUC_Type')
        
        # Find the maximum auc_test value
        max_auc_test = filtered_max_auc['auc_test'].max()
        max_row = filtered_max_auc[filtered_max_auc['auc_test'] == max_auc_test]
        max_dataset_number = max_row['dataset_number']
        
        # Add a dot at the highest auc_test value
        ax.plot(max_dataset_number-1, max_auc_test, 'ro')
        
        # ------------------------------------------------------
        
        # Find the index of the minimum 'mae_test' value for each group
        idx_min_mae = filtered_metrics.groupby('dataset_number')['mae_test'].idxmin()
        
        # Use these indices to filter the DataFrame
        filtered_min_mae = filtered_metrics.loc[idx_min_mae]
        
        # Find the minimum mae_test value
        min_mae_test = filtered_min_mae['mae_test'].min()
        
        # Find the row with the minimum mae_test value
        min_row = filtered_min_mae[filtered_min_mae['mae_test'] == min_mae_test]
        
        # Get the dataset number corresponding to the row with the minimum mae_test value
        min_dataset_number = min_row['dataset_number']

        # Add a dot at the lowest mae_test value
        ax.plot(min_dataset_number-1, min_mae_test, 'bo')
        
        # ------------------------------------------------------
        
        plt.title(f'{model}')
        plt.legend(loc='lower right')
        plt.show()