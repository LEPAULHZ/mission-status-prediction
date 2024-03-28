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