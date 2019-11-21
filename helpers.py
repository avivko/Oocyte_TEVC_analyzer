import numpy as np


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def get_closest_value_from_ordered_array(value,np_array): # Gets the first one and stops there!
    assert isinstance(value, (int,float)) and isinstance(np_array, np.ndarray), 'value or array have bad type: value is {}, array is {}'.format(type(value),np_array.dtype)
    assert np_array[0] <= value <= np_array[-1], 'value out of range of the array values: '+str(np_array[0])+' <= '+str(value)+' <= '+str(np_array[-1])
    last_value = None
    for i in np.nditer(np_array):
        if abs(np_array[0]) < abs(np_array[1]):
            if abs(i) > value:
                return float(last_value)
        elif abs(np_array[0]) > abs(np_array[1]):
            if abs(i) < value:
                return float(last_value)
        if i == value:
           return float(value)
        else:
            last_value = i
            pass
    raise ValueError('value not found in np array')


def get_index_of_unique_value(value,np_array): # returns the first one of there are a few!
    assert isinstance(value, (int,float)) and isinstance(np_array, np.ndarray), 'value or array have bad type: value is {}, array is {}'.format(type(value),np_array.dtype)
    assert np_array[0] <= value <= np_array[-1], 'value out of range of the array values: '+str(np_array[0])+' <= '+str(value)+' <= '+str(np_array[-1])
    return int(np.where(np_array == value)[0][0])
