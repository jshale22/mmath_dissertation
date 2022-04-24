import numpy as np
from all_functions.brownian_class import Brownian
import matplotlib.pyplot as plt
from all_functions.boundary_functions import function_over_region as f
import time

def simulation_bias(number_of_walks, number_of_steps_per_walk, final_time, interval_length, boundary_function):
    """
    simulation_bias runs a Brownian Motion with a bias function h on the [0,1]^2 domain and then approximates the solution to
    the heat equation with added drift (u_f) with BC u_f(x,0) = f(x)
    
    Arguments:
    number_of_walks         : Number of tracks taken at each starting point in the domain
    number_of_steps_per_walk: Number of steps each track takes until it reaches the final time T (used to determine the time step of our Brownian Motion)
    final_time              : Final time of our tracks
    interval_length         : Length between our intervals when choosing a starting point
    boundary_function       : Boundary function f(x) of our heat equation
    """
    start_simulation = time.time()
    
    # Initialising arrays
    number_of_intervals = int(np.around((1-interval_length)/interval_length))
    initial_conditions = np.around(np.array([np.linspace(interval_length,1-interval_length,number_of_intervals), \
                         np.linspace(interval_length,1-interval_length,number_of_intervals)]), decimals = 2)
    walk_endpoints = np.zeros((2, number_of_walks))
    walk_endpoints = walk_endpoints.astype(float)
    u_f = np.zeros((number_of_intervals,number_of_intervals))
    
    # Threshold for plotting as if we plot more than 10 tracks on a box it will not really provide us with much information
    plot_threshold = 10
    
    # Naming text files
    result_dir = "simulation_bias.py_results/"
    u_f_text_dir = "u_f_text_files/"
    u_f_text_name = "u_f_" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps_finaltime_" \
                    + str(final_time) + "interval_" + str(interval_length) + ".txt"
    time_text_dir = "time_text_files/"
    time_text_name = "time_for_each_point" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps_finaltime_" \
                    + str(final_time) + "interval_" + str(interval_length) + ".txt"

    with open(result_dir + time_text_dir + time_text_name, "w") as time_file:
        with open(result_dir + "h_evaluated_results.txt", "w") as h_evaluated_file:
            for j in range(number_of_intervals):
                for k in range(number_of_intervals):
                
                    start_walkloop = time.time()
                    # Defining our starting point
                    b = Brownian(x0 = [initial_conditions[0,j], initial_conditions[1,k]])
                    
                    # IF statement to determine whether the tracks will be plotted
                    if number_of_walks <= plot_threshold:
                        plt.figure()
                        for i in range(number_of_walks):
                        
                            # Running our tracks for a specified starting point and storing the endpoint values
                            walk = b.brownian_motion_bias_normal(n_steps = number_of_steps_per_walk, t = 0, T = final_time)
                            plt.plot(walk[0], walk[1], zorder=0)
                            walk_endpoints[:,i] = walk[:,number_of_steps_per_walk]
                        
                        if number_of_walks == 1:
                            plt.scatter(walk[0,number_of_steps_per_walk], walk[1,number_of_steps_per_walk], zorder=1, color = 'red', marker='x')
                        # Plotting the tracks
                        plt.xlabel("$x$")
                        plt.ylabel("$y$")
                        plt.savefig(result_dir + "random_walks_at_each_point/x0=" + str([initial_conditions[0,j],initial_conditions[1,k]]) + ".png")
                        plt.close()
                    
                    else:
                        for i in range(number_of_walks):
                        
                            # Running our tracks for a specified starting point and storing the endpoint values
                            walk = b.brownian_motion_bias_normal(n_steps = number_of_steps_per_walk, t = 0, T = final_time)
                            walk_endpoints[:,i] = walk[:,number_of_steps_per_walk]
                    
                    # Evaluating our endpoints over f(x)
                    f_evaluated = np.zeros(number_of_walks)
                    f_evaluated = boundary_function(walk_endpoints)
                    
                    # Working out the value of h(B_0), where h is the bias function for our domain, solving (1/2)Delta*h = lambda*h
                    h_b_0 = np.sin(np.pi*initial_conditions[0,j])*np.sin(np.pi*initial_conditions[1,k])
                    
                    # Calculating h(B_1)
                    h_evaluated = np.sin(np.pi*walk_endpoints[0,:])*np.sin(np.pi*walk_endpoints[1,:])
                    h_evaluated_file.write("h_evaluated at " + str([initial_conditions[0,j], initial_conditions[1,k]]) + "\n") # Write it out so that only really small values are printed?
                    h_evaluated_file.write(str(h_evaluated) + "\n")
                    
                    # Approximating u_f
                    u_f[k,j] = np.sum((h_b_0/h_evaluated)*f_evaluated)/number_of_walks # IF statement for when it is larger than some number?
                    
                    print("Finished with track loop for x0 = " + str([initial_conditions[0,j], initial_conditions[1,k]]))
                    end_walkloop = time.time()
                    time_file.write("Time for track loop for x0 = " + str([initial_conditions[0,j], initial_conditions[1,k]]) + ": {:.2f} seconds\n".format(end_walkloop - start_walkloop))
            end_simulation = time.time()
            time_file.write("\nTime for whole script: {:.2f} seconds".format(end_simulation - start_simulation))
    
    # Writing u_f over the domain to a file
    with open(result_dir + u_f_text_dir + u_f_text_name, "w") as results:
        results.write("        ")
        for i in range(number_of_intervals):
            results.write(str((i+1)/(number_of_intervals+1)) + "   ")
        results.write("\n\n")
        for i in range(number_of_intervals):
            results.write(str((i+1)/(number_of_intervals+1)) + "     ")
            for j in range(number_of_intervals):
                results.write(str(u_f[i,j]) + "   ")
            results.write("\n")

# Arguments entered into the function
n_walks = 1
n_steps = 100000
T     = 1 # Different times, find infinity time
intervals = 0.1
simulation_bias(n_walks, n_steps, T, intervals, f)
