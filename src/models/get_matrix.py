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