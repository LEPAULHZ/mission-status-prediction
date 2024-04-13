# Function to calculate imputation strategy and fill value
def calculate_imputation(column_list, feature_column, strategy):
    if not column_list:  # If the column list is empty
        return None  # Return None
    else:
        if strategy == 'mean':
            fill_value = feature_column.mean()
        elif strategy == 'median':
            fill_value = feature_column.median()
        elif strategy == 'mode':
            fill_value = feature_column.mode()[0]
        else:
            raise ValueError("Invalid imputation strategy")
        return fill_value