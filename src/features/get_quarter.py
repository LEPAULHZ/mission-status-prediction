import pandas as pd

def get_quarter(date_feature):
    """
    Convert date features to fiscal quarters.

    Parameters:
    date_feature (array-like): The array-like object containing date features in the format '%d/%m/%Y'.

    Returns:
    array-like: An array-like object indicating the fiscal quarter of each date.
                Returns 0 for Q1, 1 for Q2, 2 for Q3, 3 for Q4, and 4 for other.

    Notes:
    - The function assumes that the date feature is provided in the format '%d/%m/%Y'.
    - Invalid dates are handled by returning 'NaT'.
    - Fiscal quarters are determined based on the given conditions:
        - Q1: January 1 to April 30
        - Q2: May 1 to June 30
        - Q3: July 1 to September 30
        - Q4: October 1 to December 31
        - Other: Dates outside of the specified ranges

    Example:
    >>> dates = ['01/01/2024', '15/05/2024', '30/07/2024', '10/10/2024', '25/12/2024']
    >>> get_quarter(dates)
    [0, 1, 2, 3, 3]
    """
    date_obj = pd.to_datetime(date_feature, format='%d/%m/%Y', errors='coerce')
    # Check if any date is NaT (invalid date)
    if date_obj.isna().any():
        return 'NaT'
    
    month = date_obj.dt.month
    day = date_obj.dt.day
    is_q1 = ((month == 1) & (day >= 1)) | (month <= 4)
    is_q2 = ((month == 4) & (day >= 1)) | ((month == 5) | (month == 6))
    is_q3 = ((month == 7) & (day >= 1)) | ((month == 8) | (month == 9))
    is_q4 = ((month == 10) & (day >= 1)) | ((month == 11) | (month == 12))
    
    # Assign seasons based on conditions
    quarter = pd.Series(index=date_feature.index, dtype='string')
    quarter[is_q1] = 'Q1'  
    quarter[is_q2] = 'Q2'
    quarter[is_q3] = 'Q3' 
    quarter[is_q4] = 'Q4' 
    quarter.fillna('', inplace=True)  # Other
    return quarter