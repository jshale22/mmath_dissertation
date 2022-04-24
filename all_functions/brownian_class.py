import numpy as np
from numba import jit


@jit(nopython=True)
def bias_loop_numba(initial, increment, n_steps, h):
    """
    bias_loop_numba is used in brownian_motion_bias_numba to perform FOR loops at a faster computing speed than normal.
    Utilising the Numba module in Python in order to do this. This can't be placed inside the class.
    
    Arguments:
    initial   : Initial starting point x_0
    increment : Brownian Motion increments at each step
    n_steps   : Number of steps
    h         : Time step
    
    Returns:
        A Numpy array of 'n_steps' points
    """
    # Preallocating array
    array = np.zeros((2, n_steps+1))
    # Initial starting point
    array[:,0] = initial
    # Calculating biased Brownian Motion track
    for i in range(n_steps):
        array[:,i+1] = 1 - np.abs(np.abs(array[:, i] + increment[:,i] + (1/np.tan(np.pi*array[:,i]))*h) - 1)
    return array

class Brownian():
    """ Class constructor for Brownian  motion """
    
    def __init__(self, x0=0):
        """
        Init class
        
        Arguments:
            x0: Initial position for random walk
        """
        
        self.x0 = np.array(x0)
    
    def gen_random_normal_walk(self, n_steps = 100, t = 0, T = 1, dim = 1):
        """
        Generate motion from a Normal Distribution using FOR loops
        
        Arguments:
            n_steps: Number of steps
            t      : Initial time t
            T      : Final time T
            dim    : Dimension
        
        Returns:
            A NumPy array with 'n_steps' points
        """
        # Tests to make sure user input is correct
        assert (n_steps > 0), "Expect n_steps to be an int greater than 0"
        assert (t >= 0), "Expect initial time t to be greater than or equal to 0"
        assert (T > t), "Expect final time T to be greater than initial time t"
        assert (dim <= 3), "Expect dim to be 3 or less"
        
        # Calculating time step
        h = (T - t)/n_steps
        
        # Different output if dimension is 1
        if dim == 1:
            # Preallocating array
            walk = np.zeros(n_steps+1)
            # Initial starting point
            walk[0] = walk[0] + self.x0
            # Performing Brownian Motion algorithm
            for i in range(1, n_steps+1):
                yi = np.random.default_rng().normal(0,np.sqrt(h))
                walk[i] = walk[i-1] + yi
        else:
            # Preallocating array
            walk = np.zeros((dim, n_steps+1))
            # Initial starting point
            walk[:,0] = walk[:,0] + self.x0
            # Performing Brownian Motion algorithm
            for i in range(1, n_steps+1):
                yi = np.random.default_rng().normal(0, np.sqrt(h), dim)
                walk[:,i] = walk[:,i-1] + yi

        return walk
    
    def gen_walk_with_drift(self, n_steps = 100, t = 0, T = 1, dim = 1, a = 1):
        """
        Generate motion using gen_random_normal_walk with added drift vector a using FOR loops
        
        Arguments:
            n_steps: Number of steps
            t      : Initial time t
            T      : Final time T
            dim    : Dimension
            a      : Drift vector
        
        Returns:
            A NumPy array with 'n_steps' points
        """
        # Tests to make sure user input is correct
        assert (n_steps > 0), "Expect n_steps to be an int greater than 0"
        assert (t >= 0), "Expect initial time t to be greater than or equal to 0"
        assert (T > t), "Expect final time T to be greater than initial time t"
        assert (dim <= 3), "Expect dim to be 3 or less"
        
        # Ensuring drift vector is a numpy array
        a = np.array(a)
        # Preallocating array
        position = np.zeros((dim,n_steps+1))
        # Getting a Brownian Motion run from previous function
        walk = self.gen_random_normal_walk(n_steps, t, T, dim)
        
        # Calculating time step
        h = (T - t)/n_steps
        
        # Calculating track with the added drift
        for i in range(n_steps+1):
            position[:,i] = walk[:,i] + a*(h*i)

        return position
    
    def gen_random_normal_walk_vect(self, n_steps = 100, t = 0, T = 1, dim = 1):
        """
        Generate motion from a Normal Distribution using vectorisation
        
        Arguments:
            n_steps: Number of steps
            t      : Initial time t
            T      : Final time T
            dim    : Dimension
        
        Returns:
            A NumPy array with 'n_steps' points
        """
        # Tests to make sure user input is correct
        assert (n_steps > 0), "Expect n_steps to be an int greater than 0"
        assert (t >= 0), "Expect initial time t to be greater than or equal to 0"
        assert (T > t), "Expect final time T to be greater than initial time t"
        assert (dim <= 3), "Expect dim to be 3 or less"
        
        # Calculating time step
        h = (T - t)/n_steps
        
        # Different output if dimension is 1
        if dim == 1:
            # Preallocating array
            walk = np.zeros(n_steps+1)
            # Initial starting point
            walk[0] = self.x0
            # Performing Brownian Motion algorithm
            walk[1:] = np.random.default_rng().normal(0, np.sqrt(h), n_steps)
            walk = np.cumsum(walk)
        else:
            # Preallocating array
            walk = np.zeros((dim, n_steps+1))
            # Initial starting point
            walk[:,0] = self.x0
            # Performing Brownian Motion algorithm
            walk[:,1:] = np.random.default_rng().normal(0, np.sqrt(h), (dim,n_steps))
            walk = np.cumsum(walk, axis = 1)

        return walk

    def gen_walk_with_drift_vect(self, n_steps = 100, t = 0, T = 1, dim = 1, a = 1):
        """
        Generate motion using gen_random_normal_walk with added drift vector a using vectorisation
        
        Arguments:
            n_steps: Number of steps
            t      : Initial time t
            T      : Final time T
            dim    : Dimension
            a      : Drift vector
        
        Returns:
            A NumPy array with 'n_steps' points
        """
        # Tests to make sure user input is correct
        assert (n_steps > 0), "Expect n_steps to be an int greater than 0"
        assert (t >= 0), "Expect initial time t to be greater than or equal to 0"
        assert (T > t), "Expect final time T to be greater than initial time t"
        assert (dim <= 3), "Expect dim to be 3 or less"
        
        # Calculating time step
        h = (T - t)/n_steps
        
        # Copying drift vector across total number of steps (multiplied by time step h)
        drift = h*(np.ones((dim, n_steps+1)).T * np.array(a)).T
        # Starting point isn't affected by the drift vector
        drift[:,0] = 0
        # Cumulative sum since we are performing a*t
        drift = np.cumsum(drift, axis = 1)
        # Preallocating array
        position = np.zeros((dim,n_steps+1))
        # Getting a Brownian Motion run from previous function
        walk = self.gen_random_normal_walk_vect(n_steps, t, T, dim)
        
        # Calculating position of track at each step by adding Brownian track with drift values
        position = walk + drift

        return position
    
    def langevin_motion(self, n_steps = 100, t = 0, T = 1, initial_angle = 0, sigma = 1):
        """
        Generates motion by using Langevin Brownian Motion
        
        Arguments:
            n_steps       : Number of steps
            t             : Initial time t
            T             : Final time T
            initial_angle : The initial angle to determine the velocity vector of the track
            sigma         : Scaling parameter sigma to improve behaviour of model
        
        Returns:
            A NumPy array with 'n_steps' points
        """
        # Tests to make sure user input is correct
        assert (n_steps > 0), "Expect n_steps to be an int greater than 0"
        assert (t >= 0), "Expect initial time t to be greater than or equal to 0"
        assert (T > t), "Expect final time T to be greater than initial time t"
        
        # Calculating time step
        h = (T - t)/n_steps
        
        # Preallocating array
        walk = np.zeros((2, n_steps+1))
        # Initial starting point
        walk[:,0] = self.x0
        # 1D Brownian Motion track with initial angle which is a number between 0 and 2pi
        brownian_motion_values = np.cumsum(np.random.default_rng().normal(0, np.sqrt(h)*sigma, n_steps)) + initial_angle
        # Calculating omega value with the brownian motion track
        omega_values = np.array([np.cos(brownian_motion_values), np.sin(brownian_motion_values)])
        # Calculating Langevin Brownian Motion track as defined
        walk[:,1:] = h*omega_values
        walk = np.cumsum(walk, axis =1)
        
        return walk, omega_values, brownian_motion_values

    def brownian_motion_bias_normal(self, n_steps = 100, t = 0, T = 1):
        """
        Generates motion by using Brownian Motion with a drift bias function h.
        
        Arguments:
            n_steps: Number of steps
            t      : Initial time t
            T      : Final time T
        
        Returns:
            A NumPy array with 'n_steps' points
        """
        # Tests to make sure user input is correct
        assert (n_steps > 0), "Expect n_steps to be an int greater than 0"
        assert (t >= 0), "Expect initial time t to be greater than or equal to 0"
        assert (T > t), "Expect final time T to be greater than initial time t"
        
        # Calculating time step
        h = (T - t)/n_steps
        
        # Calculating 2D Brownian Motion track
        brownian_increments = np.random.default_rng().normal(0, np.sqrt(h), (2,n_steps))
        # Preallocating array
        walk = np.zeros((2, n_steps+1))
        # Initial starting point
        walk[:,0] = self.x0
        # Calculating biased Brownian Motion track
        for i in range(n_steps):
            walk[:,i+1] = 1 - np.abs(np.abs(walk[:, i] + brownian_increments[:,i] + (1/np.tan(np.pi*walk[:,i]))*h) - 1)
        
        return walk
    
    
    def brownian_motion_bias_numba(self, n_steps = 100, t = 0, T = 1):
        """
        Generates motion by using Brownian Motion with a drift bias function h.
        
        Arguments:
            n_steps: Number of steps
            t      : Initial time t
            T      : Final time T
        
        Returns:
            A NumPy array with 'n_steps' points
        """
        # Tests to make sure user input is correct
        assert (n_steps > 0), "Expect n_steps to be an int greater than 0"
        assert (t >= 0), "Expect initial time t to be greater than or equal to 0"
        assert (T > t), "Expect final time T to be greater than initial time t"
        
        # Calculating time step
        h = (T - t)/n_steps
        
        # Calculating 2D Brownian Motion track
        brownian_increments = np.random.default_rng().normal(0, np.sqrt(h), (2,n_steps))
        # Calculatng biased Brownain Motion track through Numba function
        walk = bias_loop_numba(self.x0, brownian_increments, n_steps, h)
        
        return walk
    
    
    def brownian_motion_with_changing_drift(self, n_steps = 100, t = 0, T = 1, a = [1,0], lambda_val = 0.5):
        """
        Generates motion by using Brownian Motion
        """
        # Tests to make sure user input is correct
        assert (n_steps > 0), "Expect n_steps to be an int greater than 0"
        assert (t >= 0), "Expect initial time t to be greater than or equal to 0"
        assert (T > t), "Expect final time T to be greater than initial time t"
        
        # Calculating time step
        time_diff = T-t
        h = time_diff/n_steps
        
        # Calculating times drift vector will change in track
        expected_no_of_time_changes = 4*np.ceil((T-t)/(1/lambda_val)).astype(int)
        time_intervals = np.cumsum(np.random.default_rng().exponential(1/lambda_val, expected_no_of_time_changes))
        # Calculating which time steps in process that the drift vector will change
        time_intervals = np.ceil((time_intervals[time_intervals <= time_diff])*n_steps).astype(int)
        
        # Creating the new drift vectors
        number_of_drift_changes = np.size(time_intervals)
        uniform_circle_points = np.random.default_rng().uniform(0, 2*np.pi, number_of_drift_changes)
        drift_vectors = np.array([np.cos(uniform_circle_points), np.sin(uniform_circle_points)])
        
        # Allocating array for drift, with initial drift vector
        drift = np.ones((2, n_steps+1))
        drift[:,0] = 0
        if number_of_drift_changes == 0:
            drift[:,1:] = h*(drift[:,1:].T * np.array(a)).T
        else:
            drift[:,1:time_intervals[0]] = h*(drift[:,1:time_intervals[0]].T * np.array(a)).T
        
        # Assigning drift vectors up to each time interval
        for i in range(number_of_drift_changes):
            if i == number_of_drift_changes-1:
                drift[:, time_intervals[i]:] = h*(drift[:, time_intervals[i]:].T * drift_vectors[:,i]).T
            else:
                drift[:, time_intervals[i]:time_intervals[i+1]] = h*(drift[:, time_intervals[i]:time_intervals[i+1]].T * drift_vectors[:,i]).T
        
        # Cumulative sum since we are performing a*t
        drift = np.cumsum(drift, axis = 1)
        
        # Preallocating array
        position = np.zeros((2,n_steps+1))
        
        # Getting a Brownian Motion run from previous function
        brownian_values = self.gen_random_normal_walk_vect(n_steps, t, T, 2)
        
        # Calculating position of track at each step by adding Brownian track with drift values
        position = brownian_values + drift
        
        return position, time_intervals, drift_vectors
        
        
        
        