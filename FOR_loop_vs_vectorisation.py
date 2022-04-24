import numpy as np
from all_functions.brownian_class import Brownian
import matplotlib.pyplot as plt
import time

"""
FOR_loop_vs_vectorisation.py is a script that tests the CPU time between using FOR loops and
vectorisation when simulating tracks of Brownian Motion with added drift.
A text file is provided which displays the different CPU times.
"""
start_script_time = time.time()
# Initial starting point, doesn't really matter what we choose
b = Brownian(x0 = [0.5,0.5])
# Testing for different number of tracks, keeping number of steps per track the same
number_of_walks = np.array([10, 100, 1000]).astype(int)
number_of_steps_per_walk = 1000
# Directory and name of output file
result_dir = "cpu_speed_comparisons/"
cpu_time_name = "cpu_time_comparison_for_loop_vs_vectoriation.txt"

with open(result_dir + cpu_time_name, "w") as results:
    for walk_number in number_of_walks:
        print("Running number of tracks: " + str(walk_number))
        results.write("Running number of tracks: " + str(walk_number) + "\n")
        start_forloops = time.time()
        # Calculating time it takes for the FOR loop version to run
        for i in range(walk_number):
            walk = b.gen_walk_with_drift(n_steps = number_of_steps_per_walk, t = 0, T = 1, dim = 2, a = [1, 0])
        end_forloops = time.time()

        start_vect = time.time()
        # Calculating time it takes for the vectorised version to run
        for i in range(walk_number):
            walk = b.gen_walk_with_drift_vect(n_steps = number_of_steps_per_walk, t = 0, T = 1, dim = 2, a = [1, 0])
        end_vect = time.time()
        
        # Printing to console and writing to output file
        print("Time for FOR loops:     {:.4f} seconds".format(end_forloops - start_forloops))
        print("Time for vectorisation: {:.4f} seconds".format(end_vect - start_vect))
        results.write("Time for FOR loops:     {:.4f} seconds \n".format(end_forloops - start_forloops))
        results.write("Time for vectorisation: {:.4f} seconds \n".format(end_vect - start_vect))
        
    end_script_time = time.time()
    print("Total time of script: {:.4f} seconds".format(end_script_time - start_script_time))
    results.write("Total time of script: {:.4f} seconds\n".format(end_script_time - start_script_time))