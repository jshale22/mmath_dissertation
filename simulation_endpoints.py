import numpy as np
from all_functions.brownian_class import Brownian
import matplotlib.pyplot as plt
from all_functions.boundary_functions import function_over_region as f
import time

"""
simulation_endpoints.py is a script that runs a Brownian Motion with a bias function h on the [0,1]^2 domain for a specific
starting point x_0. The endpoints of each track with the starting point x_0 are used to calculate the distribution of endpoints
for all tracks over the domain.
"""

start_script = time.time()

# Starting point for each track
initial_condition = np.array([0.5,0.5])

# Number of tracks and number of steps per track
number_of_walks = 10
number_of_steps_per_walk = 10

# Preallocating arrays
walk_endpoints = np.zeros((2, number_of_walks)).astype(float)
endpoint_distribution = np.zeros((10,10))

# Threshold to decide when to plot the tracks, no point displaying for more than 10
plot_threshold = 10

# Directories and file names for text files
result_dir = "simulation_endpoints.py_results/"
endpoints_text_dir = "endpoint_distribution_text_files/"
endpoints_text_name = "endpoint_distribution_" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps.txt"
time_text_dir = "time_text_files/"
time_text_name = "time_for_each_point" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps.txt"

# Initiliasing Brownian class
b = Brownian(x0 = initial_condition)

# IF statement to determine whether the tracks will be plotted
if number_of_walks <= plot_threshold:
    # Initialising plot
    plt.figure()
    for i in range(number_of_walks):
        # Using Numba version of Brownian Motion bias function
        walk = b.brownian_motion_bias_numba(n_steps = number_of_steps_per_walk, t = 0, T = 1)
        # Plotting track
        plt.plot(walk[0], walk[1])
        # Saving endpoints of each track
        walk_endpoints[:,i] = walk[:,number_of_steps_per_walk]
    
    # Saving plot
    plt.title("$x0 = $" + str([initial_condition[0], initial_condition[1]]))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig(result_dir + "random_walks_at_each_point/x0=" + str([initial_condition[0], initial_condition[1]]) + ".png")
    plt.close()
else:
    for i in range(number_of_walks):
        # Using Numba version of Brownian Motion bias function
        walk = b.brownian_motion_bias_numba(n_steps = number_of_steps_per_walk, t = 0, T = 1)
        # Saving endpoints of each track
        walk_endpoints[:,i] = walk[:,number_of_steps_per_walk]

# Calculating the distribution of the endpoints in boxes with length 0.1
# Determining which box each endpoint is in
for j in range(10):
    for k in range(10):
        endpoint_distribution[k,j] = (np.sum(np.where(np.logical_and(np.logical_and \
        (walk_endpoints[0,:] <= (j+1)*0.1, walk_endpoints[0,:] >= j*0.1), \
        np.logical_and(walk_endpoints[1,:] <= (k+1)*0.1, walk_endpoints[1,:] >= k*0.1)), 1, 0)))/number_of_walks
end_script = time.time()

# Writing time of script into file
with open(result_dir + time_text_dir + time_text_name, "w") as time_file:
    time_file.write("\nTime for whole script: {:.2f} seconds".format(end_script - start_script))

# Writing endpoint distribution to file
with open(result_dir + endpoints_text_dir + endpoints_text_name, "w") as f:
    f.write("          ")
    for i in range(10):
        f.write(str(i/10) + "-" + str((i+1)/10) + "   ")
    f.write("\n\n")
    for i in range(10):
        f.write(str(i/10) + "-" + str((i+1)/10) + "   ")
        for j in range(10):
            if j == 10:
                f.write("{:.5f}".format(endpoint_distribution[i,j]))
            else:
                f.write("{:.5f}   ".format(endpoint_distribution[i,j]))
        f.write("\n")
