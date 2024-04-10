import numpy as np

def extract_values(data):
    """
    Recursively extracts numeric and string values from nested dictionaries.

    Parameters:
    - data (dict): A nested dictionary containing various data types.

    Returns:
    - values (dict): A dictionary containing the extracted numeric and string values.

    Example:
    1. Simple example with integers and strings:
    >>> data = {'a': 1, 'b': 'hello'}
    >>> extract_values(data)
    {'a': 1, 'b': 'hello'}

    2. Nested dictionary with integers, floats, and strings:
    >>> data = {'a': 1, 'b': {'c': 2.5, 'd': 'world'}}
    >>> extract_values(data)s
    {'a': 1, 'c': 2.5, 'd': 'world'}
    """
    values = dict()
    for key, value in data.items():
        # Check if the value is an instance of various dtype
        if isinstance(value, (int, float, str, np.int64, np.float64)):
            # If yes, add the key-value pair to the values dictionary
            values[key] = value
         # Check if the value is a nested dictionary
        elif isinstance(value, dict):
            # If yes, recursively call the extract_values function on the nested dictionary
            # and update the values dictionary with the extracted values
            values.update(extract_values(value))
    return values
