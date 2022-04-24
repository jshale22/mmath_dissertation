import numpy as np
from all_functions.brownian_class import Brownian
import matplotlib.pyplot as plt

"""
brownian_motion_with_changing_drift_test.py is a script that tests the Brownian motion function
implemented in the Brownian class for changing drift vectors over time intervals. This is done
for a small number of tracks in order to plot and view the tracks in the main domain.
"""

# Assigning initial starting point of the tracks
b = Brownian(x0 = [0.5,0.5])

# Assigning number of tracks, number of steps per track and final time for the tracks
number_of_walks = 1
number_of_steps_per_walk = 10000

# Initialising plot for the Brownian Motion in D
fig1 = plt.figure()
ax1 = plt.axes()
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Running Brownian Motion function for set number of tracks
for i in range(number_of_walks):
    brownian_track, time_intervals = b.brownian_motion_with_changing_drift(n_steps = number_of_steps_per_walk, t = 0, T = 1, a = [1,0], lambda_val = 2)
    
    while (np.any(np.logical_or(brownian_track > 1, brownian_track < 0))) or np.size(time_intervals) == 0:
        brownian_track, time_intervals = b.brownian_motion_with_changing_drift(n_steps = number_of_steps_per_walk, t = 0, T = 1, a = [1,0], lambda_val = 2)
    
    # Plotting each track into main domain
    number_of_intervals = np.size(time_intervals)
    for i in range(number_of_intervals+1):
        if i == 0:
            ax1.plot(brownian_track[0, :(time_intervals[i]+1)], brownian_track[1, :(time_intervals[i]+1)], zorder=i+1)
        elif i == number_of_intervals:
            ax1.plot(brownian_track[0, time_intervals[i-1]:], brownian_track[1, time_intervals[i-1]:], zorder=i+1)
        else:
            ax1.plot(brownian_track[0, time_intervals[i-1]:(time_intervals[i]+1)], brownian_track[1, time_intervals[i-1]:(time_intervals[i]+1)], zorder=i+1)
    ax1.scatter(brownian_track[0, time_intervals], brownian_track[1, time_intervals], color="crimson", zorder=50)

# Saving figures
fig1.savefig("brownian_motion_with_changing_drift_results/brownian_motion_with_changing_drift_walk.png")
plt.show()