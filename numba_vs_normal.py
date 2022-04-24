import numpy as np
from all_functions.brownian_class import Brownian
import matplotlib.pyplot as plt
import time

"""
numba_vs_normal.py is a script that tests the CPU time when using the Numba Python module
to compute a Brownian Motion track with the bias function h. A text file is provided which
displays the different CPU times.
"""
start_script_time = time.time()
# Initial starting point, doesn't really matter what we choose
b = Brownian(x0 = [0.5, 0.5])
# Testing for different number of tracks, keeping number of steps per track the same
number_of_walks = np.array([10, 100, 1000, 10000]).astype(int)
number_of_steps_per_walk = 1000
# Directory and name of output file
result_dir = "cpu_speed_comparisons/"
cpu_time_name = "cpu_time_comparison_for_numba_vs_normal.txt"

#Initialising numba once in order to compile
start_compile_time = time.time()
walk = b.brownian_motion_bias_numba(n_steps = 1, t = 0, T = 1)
end_compile_time = time.time()

with open(result_dir + cpu_time_name, "w") as results:
    for walk_number in number_of_walks:
        print("Running number of tracks: " + str(walk_number))
        results.write("Running number of tracks: " + str(walk_number) + "\n")
        start_walkloop_for_normal = time.time()
        # Calculating time it takes for the standard version to run
        for i in range(walk_number):
            walk = b.brownian_motion_bias_normal(n_steps = number_of_steps_per_walk, t = 0, T = 1)
        end_walkloop_for_normal = time.time()

        start_walkloop_for_numba = time.time()
        # Calculating time it takes for the numba version to run
        for i in range(walk_number):
            walk = b.brownian_motion_bias_numba(n_steps = number_of_steps_per_walk, t = 0, T = 1)
        end_walkloop_for_numba = time.time()
        
        # Printing to console and writing to output file
        print("Time for normal function: {:.4f} seconds".format(end_walkloop_for_normal - start_walkloop_for_normal))
        print("Time for numba function:  {:.4f} seconds".format(end_walkloop_for_numba - start_walkloop_for_numba))
        results.write("Time for normal function: {:.4f} seconds \n".format(end_walkloop_for_normal - start_walkloop_for_normal))
        results.write("Time for numba function:  {:.4f} seconds \n".format(end_walkloop_for_numba - start_walkloop_for_numba))
    
    print("Initial compilation time for numba version: {:.4f} seconds".format(end_compile_time - start_compile_time))
    results.write("Initial compilation time for numba version: {:.4f} seconds \n".format(end_compile_time - start_compile_time))
    end_script_time = time.time()
    print("Total time of script: {:.4f} seconds".format(end_script_time - start_script_time))
    results.write("Total time of script: {:.4f} seconds\n".format(end_script_time - start_script_time))
