def mission_percent(df, column_name):
    # Count total, success, and failure missions directly
    mission_count = df[column_name].value_counts().sort_index()
    success_counts = df[df['Status Mission'] == 'Success'].groupby(column_name).size() 
    failure_counts = df[df['Status Mission'] != 'Success'].groupby(column_name).size() 

    # Compute percentages
    success_percent = (success_counts / mission_count * 100).reindex(mission_count.index).fillna(0)
    failure_percent = (failure_counts / mission_count * 100).reindex(mission_count.index).fillna(0)
    
    return success_percent, failure_percent

