import pandas as pd

# Define a function to split and clean location columns
def split_and_clean_location(df, num_commas):
    # Filter rows with the specified number of commas
    filtered_df = df[df['Location'].str.count(',') == num_commas]
    
    # Split the text into separate columns
    split_df = filtered_df['Location'].str.split(',', expand=True)
    
    # Add empty columns if necessary
    if num_commas == 2:
        split_df.insert(num_commas, f'Empty Column {num_commas}', 'missing')
    elif num_commas == 1:
        split_df.insert(num_commas-1, f'Empty Column {num_commas-1}', 'missing')
        split_df.insert(num_commas+1, f'Empty Column {num_commas+1}', 'missing')

    # Rename the columns
    column_names = ['Pad', 'Center', 'State', 'Country']
    split_df.columns = column_names[:split_df.shape[1]]
    
    # Strip whitespace from each column
    split_df = split_df.apply(lambda x: x.str.strip())
    
    return split_df

# Define a function to replace specific values in columns
def replace_values(df):
    df['State'] = df['State'].str.replace('Maranh?œo', 'Maranhao')
    df['Center'] = df['Center'].str.replace('Alc?›ntara Launch Center', 'Alcantara Space Center')
    df['Center'] = df['Center'].str.replace('M?\x81hia Peninsula', 'Mahia Peninsula')
    return df
