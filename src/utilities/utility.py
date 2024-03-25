import pandas as pd

def accuracy_score(actual, predict):
    """
    Calculate the accuracy from two output arrays.

    Parameters:
    actual (array-like): The array containing the actual labels or values.
    predict (array-like): The array containing the predicted labels or values.

    Returns:
    float: The accuracy score, calculated as the proportion of correct predictions
           to the total number of observations.

    Note:
    The arrays `actual` and `predict` should have the same length.

    Example:
    >>> actual = [1, 0, 1, 1, 0]
    >>> predict = [1, 1, 0, 1, 0]
    >>> accuracy_score(actual, predict)
    0.6
    """
    correct_predictions = sum(actual==predict)
    total_observations = len(actual) | len(predict)
    
    return correct_predictions/total_observations
#____________________________________________________________________________#    
def decipher_confusion_matrix(confusion_matrix):
    """
    Deciphers a confusion matrix and provides relevant metrics.

    Parameters:
    confusion_matrix (array-like): A 2x2 confusion matrix representing the counts of true negatives,
                                    false positives, false negatives, and true positives, respectively.

    Returns:
    tuple: A tuple containing the following metrics:
           - True negatives (TN)
           - False positives (FP)
           - False negatives (FN)
           - True positives (TP)
           - Precision
           - Recall
           - F1 score

    Notes:
    - The confusion matrix should be provided as a 2x2 array-like object.
    - The layout of the confusion matrix should follow the convention:
        [[TN, FP],
         [FN, TP]]

    Example:
    >>> confusion_matrix = [[50, 10],
                            [5, 35]]
    >>> decipher_confusion_matrix(confusion_matrix)
    (50, 10, 5, 35, 0.7777777777777778, 0.875, 0.8235294117647058)
    """
    true_negatives = confusion_matrix[0, 0]
    false_positives = confusion_matrix[0, 1]
    false_negatives = confusion_matrix[1, 0]
    true_positives = confusion_matrix[1, 1]
    accuracy = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)
    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    f1 = 2*(precision*recall)/(precision+recall)
    
    
    return true_negatives, false_positives,false_negatives, true_positives, accuracy, precision, recall, f1

#____________________________________________________________________________#
def is_weekend(date_feature):
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
    >>> is_weekend(dates)
    [0, 1, 0, 0, 0]
    """
    date_obj = pd.to_datetime(date_feature, format='%d/%m/%Y', errors='coerce')
    # Check if any date is NaT (invalid date)
    if date_obj.isna().any():
        return 'NaT'
    weekday = date_obj.dt.weekday
    return (weekday >= 5).astype(int) # Saturday (5) or Sunday (6)

#____________________________________________________________________________#
def get_season(date_feature):
    """
    Convert date features to seasonality.

    Parameters:
    date_feature (array-like): The array-like object containing date features in the format '%d/%m/%Y'.

    Returns:
    array-like: An array-like object indicating the seasonality of each date.
                Returns 0 for winter, 1 for spring, 2 for summer, 3 for autumn, and 4 for other.

    Notes:
    - The function assumes that the date feature is provided in the format '%d/%m/%Y'.
    - Invalid dates are handled by returning 'NaT'.
    - Seasons are determined based on the given conditions:
        - Winter: December 1 to February 28/29
        - Spring: March 1 to May 31
        - Summer: June 1 to August 31
        - Autumn: September 1 to November 30
        - Other: Dates outside of the specified ranges

    Example:
    >>> dates = ['01/01/2024', '05/03/2024', '10/05/2024', '15/07/2024', '20/09/2024']
    >>> get_season(dates)
    ['Winter', 'Spring', 'Spring', 'Summer', 'Autumn']
    """
    date_obj = pd.to_datetime(date_feature, format='%d/%m/%Y', errors='coerce')
    # Check if any date is NaT (invalid date)
    if date_obj.isna().any():
        return 'NaT'
    
    month = date_obj.dt.month
    day = date_obj.dt.day
    is_winter = ((month == 12) & (day >= 1)) | (month <= 2)
    is_spring = ((month == 3) & (day >= 1)) | ((month == 4) | (month == 5))
    is_summer = ((month == 6) & (day >= 1)) | ((month == 7) | (month == 8))
    is_autumn = ((month == 9) & (day >= 1)) | ((month == 10) | (month == 11))
    
    # Assign seasons based on conditions
    season = pd.Series(index=date_feature.index, dtype='int64')
    season[is_winter] = 'Winter'  
    season[is_spring] = 'Spring'
    season[is_summer] = 'Summer' 
    season[is_autumn] = 'Autumn' 
    season.fillna('', inplace=True)  # Other
    return season

#____________________________________________________________________________#
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
    quarter = pd.Series(index=date_feature.index, dtype='int64')
    quarter[is_q1] = 'Q1'  
    quarter[is_q2] = 'Q2'
    quarter[is_q3] = 'Q3' 
    quarter[is_q4] = 'Q4' 
    quarter.fillna('', inplace=True)  # Other
    return quarter