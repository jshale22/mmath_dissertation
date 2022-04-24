import numpy as np
from all_functions.brownian_class import Brownian
import matplotlib.pyplot as plt

"""
langevin_brownian_test.py is a script that tests the Langevin Brownian motion function
implemented in the Brownian class. This is done for a small number of tracks in order to
plot and view the tracks in the main domain, as well as in the angular domain, i.e. the 
1D Brownian motion tracks on the circle.
"""

# Assigning initial starting point of the tracks
b = Brownian(x0 = [0.5,0.5])

# Assigning number of tracks, number of steps per track and final time for the tracks
number_of_walks = 5
number_of_steps_per_walk = 10000
final_time = 10

# Initialising plot for the 1D Brownian motion along the circle as time passes
fig1 = plt.figure()
ax1 = plt.axes(projection = '3d')
ax1.set_zlabel("Time")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
time = np.linspace(1/number_of_steps_per_walk, final_time, number_of_steps_per_walk)

# Initialising plot for the Langevin Brownian Motion in R^2
fig2 = plt.figure()
ax2 = plt.axes()
ax2.set_xlabel("x")
ax2.set_ylabel("y")
# Running Langevin Brownian Motion function for set number of tracks
for i in range(number_of_walks):
    walk, omega, _ = b.langevin_motion(n_steps = number_of_steps_per_walk, t = 0, T = final_time)
    
    # Plotting each track into main domain
    ax2.plot(walk[0], walk[1])
    # Plotting each 1D Brownian motion along the circle into 3D domain
    ax1.plot3D(omega[0],omega[1],time)

# Saving figures
fig1.savefig("langevin_brownian_motion_results/langevin_brownian_motion_walk.png")
fig2.savefig("langevin_brownian_motion_results/langevin_brownian_motion_omega.png")
plt.show()