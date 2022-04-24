import numpy as np
import matplotlib.pyplot as plt


def histogram_plot_for_uf(number_of_walks, number_of_steps_per_walk, final_time, interval_length, show_plot = True):
    """
    histogram_plot_for_uf plots our function u_f that was solved numerically over the domain [0,1]^2. This is u_f as
    defined in the heat equation with added drift.
    
    Arguments:
    number_of_walks         : Number of tracks taken at each starting point in the domain
    number_of_steps_per_walk: Number of steps each track takes until it reaches the final time T (needed to find the text file we created)
    final_time              : Final time of our tracks
    interval_length         : Length between our intervals when choosing a starting point
    show_plot               : True/False Boolean to check whether the plot is shown when the function is called
    """
    
    # Initialising u_f array to store the text values into
    number_of_intervals = int(np.around((1-interval_length)/interval_length))
    u_f = np.zeros((number_of_intervals,number_of_intervals))
    
    # Defining directory and text file names
    text_file_dir = "simulation_bias.py_results/no_overwritten_files/"
    text_file_name = "u_f_" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps_finaltime_" \
                    + str(final_time) + "interval_" + str(interval_length) + ".txt"
    histogram_dir = "histograms_of_uf/"
    histogram_name = "histogram_of_uf_" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps_finaltime_" \
                    + str(final_time) + "interval_" + str(interval_length) + ".png"
    
    # Taking u_f values from text file into an np.array
    with open(text_file_dir + text_file_name) as f:
        for i, line in enumerate(f):
            data = line.split()
            if i == 0 or i == 1:
                continue
            else:
                u_f[i-2] = np.array(data[1:])
    
    # Normalising u_f
    u_f = u_f/np.max(u_f)
    
    # Plotting u_f over [0,1]^2 domain
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    x = np.linspace(1-interval_length, interval_length, number_of_intervals)
    y = np.linspace(1-interval_length, interval_length, number_of_intervals)
    x, y = np.meshgrid(x - (interval_length/2),y - (interval_length/2))
    x, y = x.ravel(), y.ravel()
    z = u_f.ravel()
    width = interval_length
    depth = interval_length
    bottom = np.zeros_like(z)
    ax.bar3d(x,y,bottom, width, depth, z, shade = True, color = "b")
    
    # Plotting the bias function h over [0,1]^2 domain
    h_x = np.outer(np.linspace(0,1,100), np.ones(100))
    h_y = h_x.copy().T
    h_z = (np.sin(np.pi*h_x))*(np.sin(np.pi*h_y))
    ax.plot_surface(h_x, h_y,h_z, color = "g",alpha = 0.25)
    
    # Plotting score zone quadrant, as defined with our boundary function f(x)
    score_zone_x = np.array([0])
    score_zone_y = np.array([0])
    score_zone_z = np.array([1])
    score_zone_width = 0.5
    score_zone_depth = 0.5
    score_zone_bottom = np.zeros_like(score_zone_z)
    ax.bar3d(score_zone_x,score_zone_y,score_zone_bottom, score_zone_width, score_zone_depth, score_zone_z, alpha = 0.25, color = "r", shade = True)
    
    # Labelling of plot
    ax.set_title("Histogram displaying $u_f$ over $D$")
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("$u_f$")
    
    # Saving the figure
    fig.savefig(histogram_dir + histogram_name)
    # Showing histogram plot through console unless user specifies not to
    if show_plot:
        plt.show()

def histogram_plot_endpoints_for_uf(number_of_walks, number_of_steps_per_walk, show_plot = True):
    """
    histogram_plot_endpoints_for_uf plots the distribution of endpoints over the domain [0,1]^2
    for a starting point x_0. This is u_f as defined in the heat equation with added drift.
    Currently only have it for the case that x_0 = [0.5,0.5]^T.
    
    Arguments:
    number_of_walks         : Number of tracks taken at each starting point in the domain
    number_of_steps_per_walk: Number of steps each track takes until it reaches the final time T (needed to find the text file we created)
    show_plot               : True/False Boolean to check whether the plot is shown when the function is called
    """
    # Preallocating array for distribution of endpoints in the main domain, currently only
    # for intervals of 0.1
    endpoint_distribution = np.zeros((10,10))

    # Defining directory and text file names
    text_file_dir = "simulation_endpoints.py_results/no_overwritten_files/"
    text_file_name = "endpoint_distribution_" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps.txt"
    histogram_dir = "histograms_of_endpoints/"
    histogram_name = "histogram_of_endpoint_distribution_" + str(number_of_walks) + "walkperpoint_" + str(number_of_steps_per_walk) + "steps.png"

    # Taking the endpoint values from text file into a np.array
    with open(text_file_dir + text_file_name) as f:
        for i, line in enumerate(f):
            data = line.split()
            if i == 0 or i == 1:
                continue
            else:
                endpoint_distribution[i-2] = np.array(data[1:])
    
    # Normalising the distribution
    endpoint_distribution = endpoint_distribution/np.max(endpoint_distribution)
    
    # Plotting endpoint distribution over [0,1]^2 domain
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    x = np.linspace(0.05, 0.95, 10)
    y = np.linspace(0.05, 0.95, 10)
    x, y = np.meshgrid(x - 0.05,y - 0.05)
    x, y = x.ravel(), y.ravel()
    z = endpoint_distribution.ravel()
    width = 0.1
    depth = 0.1
    bottom = np.zeros_like(z)
    ax.bar3d(x,y,bottom, width, depth, z, shade = True)

    # Plotting the bias function h over [0,1]^2 domain
    h_x = np.outer(np.linspace(0,1,100), np.ones(100))
    h_y = h_x.copy().T
    h_z = (np.sin(np.pi*h_x))*(np.sin(np.pi*h_y))
    ax.plot_surface(h_x, h_y,h_z, alpha = 0.25)

    # Labelling of plot
    ax.set_title("Histogram displaying endpoint_distribution over [0,1]^2 box with " + str(number_of_walks) + " walks and " + str(number_of_steps_per_walk) + " steps per walk")
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("endpoint_distribution")
    
    # Saving the figure
    plt.savefig(histogram_dir + histogram_name)
    # Showing histogram plot through console unless user specifies not to
    if show_plot:
        plt.show()

# Arguments entered into the function
n_walks   = 10000
n_steps   = 10000
T         = 0.175
intervals = 0.1
# Note that if you want to see both plots when calling to console. You must close the first to see the second, as plt.show() pauses the script
histogram_plot_for_uf(n_walks, n_steps, T, intervals)
histogram_plot_endpoints_for_uf(n_walks, n_steps)
