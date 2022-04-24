import numpy as np
from all_functions.brownian_class import Brownian
import matplotlib.pyplot as plt
from all_functions.boundary_functions import function_over_region as f
import time

"""
simulation.py runs a Brownian Motion with added drift on the [0,1]^2 domain and then approximates the solution to
the heat equation with added drift (u_f) with BC u_f(x,0) = f(x). Currently takes very long to run for large number
of tracks and steps per track. See simulation_bias.py to run for larger values.
"""

start_script = time.time()

# Creating all starting points in domain
initial_conditions = np.around(np.array([np.linspace(0.1,0.9,9), np.linspace(0.1,0.9,9)]), decimals = 1)

# Number of tracks and number of steps per track
number_of_walks = 1
number_of_steps_per_walk = 100000

# Preallocating arrays
walk_endpoints = np.zeros((2, number_of_walks))
u_f = np.zeros((9,9))

# Threshold to decide when to plot the tracks, no point displaying for more than 10
plot_threshold = 10

# Directories and file names for text files
result_dir = "simulation.py_results/"
u_f_text_dir = "u_f_text_files/"
u_f_text_name = "u_f_" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps.txt"
time_text_dir = "time_text_files/"
time_text_name = "time_for_each_point" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps.txt"

with open(result_dir + time_text_dir + time_text_name, "w") as time_file:
    for j in range(9):
        for k in range(9):
            start_walkloop = time.time()
            # Defining out starting point
            b = Brownian(x0 = [initial_conditions[0,j], initial_conditions[1,k]])
            
            # IF statement to determine whether the tracks will be plotted
            if number_of_walks <= plot_threshold:
                plt.figure()
                for i in range(number_of_walks):
                    # Running our tracks for a specified starting point
                    walk = b.gen_walk_with_drift_vect(n_steps = number_of_steps_per_walk, t = 0, T = 1, dim = 2, a = [1, 0])
                    
                    # If the track leaves the domain then we must rerun it until it stays inside
                    while (np.any(np.logical_or(walk > 1, walk < 0))):
                    
                        walk = b.gen_walk_with_drift_vect(n_steps = number_of_steps_per_walk, t = 0, T = 1, dim = 2, a = [1, 0])
                    
                    # Plotting the track
                    plt.plot(walk[0], walk[1], zorder=0)
                    # Storing endpoint of each track
                    walk_endpoints[:,i] = walk[:,number_of_steps_per_walk]
                
                if number_of_walks == 1:
                    plt.scatter(walk[0,number_of_steps_per_walk], walk[1,number_of_steps_per_walk], zorder=1, color = 'red', marker='x')
                
                # Plotting the tracks and saving
                plt.xlabel("$x$")
                plt.ylabel("$y$")
                plt.savefig(result_dir + "random_walks_at_each_point/x0=" + str([initial_conditions[0,j],initial_conditions[1,k]]) + ".png")
                plt.close()
                
            else:
                for i in range(number_of_walks):
                    # Running our tracks for a specified starting point
                    walk = b.gen_walk_with_drift_vect(n_steps = number_of_steps_per_walk, t = 0, T = 1, dim = 2, a = [1, 0])
                    
                    # If the track leaves the domain then we must rerun it until it stays inside
                    while (np.any(np.logical_or(walk > 1, walk < 0))):
                    
                        walk = b.gen_walk_with_drift_vect(n_steps = number_of_steps_per_walk, t = 0, T = 1, dim = 2, a = [1, 0])
                    # Storing endpoint of each track
                    walk_endpoints[:,i] = walk[:,number_of_steps_per_walk]
            
            # Preallocating array
            f_evaluated = np.zeros(number_of_walks)
            # Evaluating our endpoints with the function f
            f_evaluated = f(walk_endpoints)
            # Calculating u_f at each starting point
            u_f[k,j] = np.sum(f_evaluated)/number_of_walks
            
            print("Finished with walk loop for x0 = " + str([initial_conditions[0,j], initial_conditions[1,k]]))
            end_walkloop = time.time()
            time_file.write("Time for walk loop for x0 = " + str([initial_conditions[0,j], initial_conditions[1,k]]) + ": {:.2f} seconds\n".format(end_walkloop - start_walkloop))
    end_script = time.time()
    time_file.write("\nTime for whole script: {:.2f} seconds".format(end_script - start_script))


# Writing u_f to a text file
with open(result_dir + u_f_text_dir + u_f_text_name, "w") as f:
    f.write("        ")
    for i in range(9):
        f.write(str((i+1)/10) + "   ")
    f.write("\n\n")
    for i in range(9):
        f.write(str((i+1)/10) + "     ")
        for j in range(9):
            if j == 10:
                f.write(str(u_f[i,j]))
            else:
                f.write(str(u_f[i,j]) + "   ")
        f.write("\n")
