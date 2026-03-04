import numpy as np

def adjust_shape(arr: np.ndarray, expected_order=2, arr_name="arr"):
    '''
    If `arr` has `expected_order` indices, adjusts its shape to have 
    `expected_order + 1` indices.
    For example, if `expected_order=2` and the shape of `arr` is (N, M), the adjustment 
    changes the shape to (1, N, M).
    '''
    num_indices = len(arr.shape) 
    if num_indices == expected_order:
        arr = arr.reshape(1, *arr.shape)
    elif num_indices != (expected_order + 1):
        order1, order2 = expected_order, expected_order + 1
        raise Exception(f"`len({arr_name}.shape)` é {len(arr.shape)}, mas deveria ser {order1} ou {order2}.")

    return arr

def remove_cells_out_of_bounds(data, many_layers=False):
    '''
    Given the grid data array `data`, with shape (num_lines, num_cols, ...),
    removes the cells that are outside the grid. If `data` has multiple layers, the parameter
    `many_layers` should be set to `True`.
    '''
    if many_layers:
        return  data[:, :-2, :-2]
    else:
        return  data[:-2, :-2]
    
def simplify_arr_shape(arr: np.array):
    if arr.shape[0] == 1:
        arr = arr.reshape(*arr.shape[1:])
    return arr
