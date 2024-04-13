import pandas as pd

def get_weekend(date_feature):
    """
    Convert date features to weekend & weekday.

    Parameters:
    date_feature (array-like): The array-like object containing date features in the format '%d/%m/%Y'.

    Returns:
    array-like: An array-like object indicating whether each date falls on a weekend (Saturday or Sunday).
                Returns 1 for weekend and 0 for weekday.

    Notes:
    - The function assumes that the date feature is provided in the format '%d/%m/%Y'.
    - Invalid dates are handled by returning 'NaT'.
    - Saturday is considered as 5 and Sunday as 6 in the weekday numbering.

    Example:
    >>> dates = ['01/01/2024', '05/01/2024', '10/01/2024', '15/01/2024', '20/01/2024']
    >>> get_weekend(dates)
    [0, 1, 0, 0, 0]
    """
    date_obj = pd.to_datetime(date_feature, format='%d/%m/%Y', errors='coerce')
    # Check if any date is NaT (invalid date)
    if date_obj.isna().any():
        return 'NaT'
    weekday = date_obj.dt.weekday
    return (weekday >= 5) # Saturday (5) or Sunday (6)