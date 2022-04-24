import numpy as np

def function_over_region(x):
    """
    Boundary condition u_f(x,0) = f(x) where f(x) = 1 if in specified region of [0,1]^2. Region is quarter of the box [0,1]^2.
    
    Arguments:
        x: 2-D vector which is a point in [0,1]^2
    
    Returns:
        1 or 0 dependent on whether x is in a specified region of [0,1]^2
    """
    
    assert (type(x) == np.ndarray and np.size(x,0) == 2), "Expect a NumPy array with 2-D points."
    
    f = np.where(np.logical_and(np.logical_and(x[0,:] <= 0.5, x[0,:] >= 0), np.logical_and(x[1,:] <= 0.5, x[1,:] >= 0)), 1, 0)
    
    return f
