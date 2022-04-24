import numpy as np
from all_functions.brownian_class import Brownian
import matplotlib.pyplot as plt
from all_functions.boundary_functions import function_over_region as f
import time
from tqdm import tqdm

def calculate_mesh(angular_N_c, main_N_c):
    """
    discretise_domain discretises the domains specified in Langevin Brownian Motion. In particular,
    this includes the box [0,1]^2 and the angular domain from [0,2pi). This provides us with the main
    in and out vectors we need to calculate our matrix after running multiple simulations.
    
    Arguments:
    angular_N_c : Specified number of intervals for Angular domain ([0,2pi))
    main_N_c    : Specified number of intervals for Main domain ([0,1]^2 box)
    """
    
    # First do for main domain, let's make the mesh go from left to right, bottom to top. Then boundaries are specified
    # starting at the top and going clockwise.
    
    # Calculating number boxes
    number_of_boxes        = main_N_c**2
    
    # Calculating which number associates to which box, should I be preallocating here?
    corner_box_mesh_numbers = np.array((1, main_N_c, (number_of_boxes+1) - main_N_c, number_of_boxes))
    edge_box_mesh_numbers   = np.concatenate((np.arange(2, corner_box_mesh_numbers[1]), \
                              np.arange(1+corner_box_mesh_numbers[1], corner_box_mesh_numbers[2], main_N_c), \
                              np.arange(2*corner_box_mesh_numbers[1], corner_box_mesh_numbers[3], main_N_c), \
                              np.arange(1+corner_box_mesh_numbers[2], corner_box_mesh_numbers[3])))
    # Just need to know which numbers have been used and then give the leftover numbers to centre boxes
    used_box_mesh_numbers   = np.concatenate((corner_box_mesh_numbers, edge_box_mesh_numbers))
    centre_box_mesh_numbers = np.arange(1, number_of_boxes+1)
    centre_box_mesh_numbers = centre_box_mesh_numbers[np.all((centre_box_mesh_numbers != used_box_mesh_numbers[:,None]),axis=0)]
    
    # Putting into list
    main_mesh_numbers = [corner_box_mesh_numbers,edge_box_mesh_numbers, centre_box_mesh_numbers]
    
    # For angular domain, this is more straightforward as each segment has 2 boundaries other than the edge ones.
    edge_angular_mesh_numbers = np.array((1, angular_N_c))
    centre_angular_mesh_numbers = np.arange(1, angular_N_c)
    centre_angular_mesh_numbers = centre_angular_mesh_numbers[np.all((centre_angular_mesh_numbers != edge_angular_mesh_numbers[:,None]),axis=0)]
    
    # Putting into list
    angular_mesh_numbers = [edge_angular_mesh_numbers, centre_angular_mesh_numbers]
    
    return main_mesh_numbers, angular_mesh_numbers
    
    
def langevin_motion_sims(number_of_tracks, number_of_steps, initial_time, final_time, sig = 1):
    """
    langevin_motion_sims simulates Langevin Brownian Motion for a specified number of tracks and steps per track.
    
    Arguments:
    number_of_tracks : Specified number of tracks to be used to approximate our matrix
    number_of_steps  : Specified number of steps per track
    initial_time     : Initial time of particle track
    final_time       : Final time of particle track
    sig              : Scaling parameter sigma to improve behaviour of model
    """
    # Preallocating storage
    all_main_tracks = np.zeros((number_of_tracks, 2, number_of_steps+1))
    all_angular_tracks = np.zeros((number_of_tracks, number_of_steps+1))
    
    # Calling Brownian class to perform Langevin for n_steps
    for i in tqdm(range(number_of_tracks)):
        # Randomly chosen starting point for each track
        starting_point = np.random.default_rng().uniform(0, 1, 2)
        # Randomly chosen initial angle for each track
        starting_angle = np.random.default_rng().uniform(0, 2*np.pi)
        
        # Initialising the Brownian class for some starting point
        b = Brownian(x0 = starting_point)
        
        # Calling Langevin Brownian Motion function. Outputs movement in domain as well as the 1D Brownian Motion tracks
        # Initial time is 0 and final time is 5 at the moment.
        # Also outputs values of omega on the circle, this isn't taken.
        # TODO - sort out the angular tracks once main tracks works
        all_main_tracks[i,:,:], _, all_angular_tracks[i,1:] = b.langevin_motion(n_steps = number_of_steps, t = initial_time, T = final_time, initial_angle = starting_angle, sigma = sig)
    
    return all_main_tracks, all_angular_tracks

def find_box(intervals, starting_point, N_c):
    """
    find_box works out which box the track is in for the main domain.
    
    Arguments:
    intervals      : Intervals between the boxes in the discretised space
    starting_point : Starting point of the track
    N_c            : Specified number of intervals for Main domain
    """
    
    # Finding which column and row the starting point is in so that I can calculate which box number it is in
    column_num = np.searchsorted(intervals, starting_point[0])
    row_num = np.searchsorted(intervals, starting_point[1])
    
    # Calculating box number as defined initially
    box_number = column_num + (row_num-1)*N_c


    return box_number, column_num, row_num

def find_surrounding_boxes(mesh_numbers, N_c, box_number):
    """
    find_surrounding_boxes calculates which boxes are connected to the current box our track is in.
    
    Arguments:
    mesh_numbers : Numbers representing the box numbers in the discretised domain
    N_c          : Specified number of intervals for Main domain
    box_number   : Current box the track is in
    """
    
    # If the box is a centre box
    if np.any(mesh_numbers[2] == box_number):
        surrounding_boxes = np.concatenate((np.arange(box_number - N_c - 1, box_number - N_c + 2), \
                            np.array((box_number - 1, box_number + 1)),                            \
                            np.arange(box_number + N_c - 1, box_number + N_c + 2)))

    # If the box is an edge box, 4 different types of edge boxes
    elif np.any(mesh_numbers[1] == box_number):
        if box_number < N_c:
            surrounding_boxes = np.concatenate((np.array((box_number - 1, box_number + 1)), np.arange(box_number + N_c - 1, box_number + N_c + 2)))
        elif np.mod(box_number, N_c) == 0:
            surrounding_boxes = np.array((box_number - N_c - 1, box_number - N_c, box_number - 1, box_number + N_c - 1, box_number + N_c))
        elif np.mod(box_number - 1, N_c) == 0:
            surrounding_boxes = np.array((box_number - N_c, box_number - N_c + 1, box_number + 1, box_number + N_c, box_number + N_c + 1))
        else:
            surrounding_boxes = np.concatenate((np.arange(box_number - N_c - 1, box_number - N_c + 2), np.array((box_number - 1, box_number + 1))))
    
    # If the box is a corner box, 4 corner boxes
    else:
        if box_number > N_c:
            if np.mod(box_number, N_c) == 0:
                surrounding_boxes = np.array((box_number - N_c - 1, box_number - N_c, box_number - 1))
            else:
                surrounding_boxes = np.array((box_number - N_c, box_number - N_c + 1, box_number + 1))
        else:
            if box_number == N_c:
                surrounding_boxes = np.array((box_number - 1, box_number + N_c - 1, box_number + N_c))
            else:
                surrounding_boxes = np.array((box_number + 1, box_number + N_c, box_number + N_c + 1))
    
    return surrounding_boxes
    
def find_J_vector_index(box_number, adj_boxes, row_num, N_c):
    """
    find_J_vector_index finds the index of the J_in and the J_out vector that we need to score as the track moves in our
    domain.
    
    Arguments:
    box_number : Current box the track is in
    adj_boxes  : The adjacent boxes to the current box
    row_num    : The row that the current box is in
    N_c        : Specified number of intervals for Main domain
    """
    
    # If the box is a centre box
    if np.size(adj_boxes) == 8:
        J_index = 3*2 + 5*(N_c - 5 + 2*row_num) + 8*((N_c-2)*(row_num-2)+np.mod(box_number,N_c)-2) # Simplified computation by moving any multiplied subtractions outside

    # If the box is an edge box
    elif np.size(adj_boxes) == 5:
        if box_number < N_c:
            J_index = 3 + 5*(box_number - 2)
        elif box_number > N_c**2 - (N_c - 1):
            J_index = 3*3 + 5*(3*N_c + np.mod(box_number, N_c) - 8) + 8*((N_c - 2)**2)
        else:
            dummy_mod = np.mod(box_number, N_c)
            J_index = 3*2 + 5*(N_c - 6 + (1-dummy_mod) + 2*row_num) + 8*((N_c-2)*(row_num - (1+dummy_mod)))

    # If the box is a corner box
    else:
        if box_number == 1:
            J_index = 0
        elif box_number == N_c:
            J_index = 3 + 5*(N_c - 2)
        elif box_number == N_c**2:
            J_index = 3*3 + 5*(4*(N_c-2)) + 8*((N_c-2)**2)
        else:
            J_index = 3*2 + 5*(3*(N_c-2)) + 8*((N_c-2)**2)
    
    return J_index

def tracking_main_domain(mesh_numbers, main_tracks, N_c):
    """
    tracking_main_domain counts the number of times the track leaves a box and enters a new one in the main domain.
    It then places them in the J_in and J_out vectors to show the number of times each occurs.
    
    Arguments:
    mesh_numbers : Numbers representing the box numbers in the discretised domain
    main_tracks  : Langevin Brownian motion tracks
    N_c          : Specified number of intervals for Main domain
    """
    # Making the intial scoring to be 0.5 instead of 0 as to not get 0 divisions in the J_in and J_out vectors
    # Helps when calculating weight in biased simulation
    J_in = np.ones(np.size(mesh_numbers[0])*3 + np.size(mesh_numbers[1])*5 + np.size(mesh_numbers[2])*8)*0.5
    J_out = np.ones(np.size(mesh_numbers[0])*3 + np.size(mesh_numbers[1])*5 + np.size(mesh_numbers[2])*8)*0.5
    # Need intervals to calculate which box the track is moving to at each step
    intervals = np.linspace(0,1,N_c+1)
    
    # Going over each track until it leaves the box
    for track_num in range(np.size(main_tracks, axis=0)):
        # Need this to account for when it enters a box at the beginning, otherwise we get row sums > 1 in probability matrix
        left_starting_box = False
        # Calculating where the track starts in the mesh
        box_number, column_num, row_num = find_box(intervals, main_tracks[track_num,:,0], N_c)
        # Finding it's adjacent boxes
        adj_boxes = find_surrounding_boxes(mesh_numbers, N_c, box_number)
        # Finding what index in the J_out vector index
        J_out_vector_index = find_J_vector_index(box_number, adj_boxes, row_num, N_c)
        # Running each step of the track
        for path_index in range(1,np.size(main_tracks, axis=2)):
            # If the track leaves the box then move to the next
            if np.any(main_tracks[track_num,:,path_index] > 1) or np.any(main_tracks[track_num,:,path_index] < 0):
                break
            # Calculating where the track is for the next step
            next_box_number, next_column_num, next_row_num = find_box(intervals, main_tracks[track_num,:,path_index], N_c)
            
            # If the box number is the same then there is nothing to be done and we move to the next step
            if next_box_number != box_number:
                # Calculating adjacent boxes for next box in track
                next_adj_boxes = find_surrounding_boxes(mesh_numbers, N_c, next_box_number)
                
                # Calculating the J_in vector index
                J_in_vector_index = find_J_vector_index(next_box_number, next_adj_boxes, next_row_num, N_c)

                # Finding which position the next and previous box is in for the Numpy array in order to add this to the J_out and J_in
                # vector indexes
                to_which_box = np.where(adj_boxes == next_box_number)
                from_which_box = np.where(next_adj_boxes == box_number)
                
                # Checking if we have moved out of the initial starting box yet as we do not want to account for the J_out value otherwise
                if left_starting_box:
                    # Scoring for each time we have movement from one box to another
                    J_out[J_out_vector_index + to_which_box[0]] = J_out[J_out_vector_index + to_which_box[0]] + 1
                    J_in[J_in_vector_index + from_which_box[0]] = J_in[J_in_vector_index + from_which_box[0]] + 1
                else:
                    J_in[J_in_vector_index + from_which_box[0]] = J_in[J_in_vector_index + from_which_box[0]] + 1
                    left_starting_box = True
                
                # Preparing for next step
                column_num = next_column_num
                row_num = next_row_num
                box_number = next_box_number
                adj_boxes = next_adj_boxes
                J_out_vector_index = J_in_vector_index
    return J_in, J_out
        
def compute_probability_matrix(J_in, J_out, mesh_numbers,N_c):
    """
    compute_probability_matrix calculates a probability matrix which determines the probability
    that a particle leaves box i and enters box j.
    
    Arguments:
    J_in         : A vector containing the number of times a particle moved from box i into box j
    J_out        : A vector containing the number of times a particle moved from box i out to box j
    mesh_numbers : Numbers representing the box numbers in the discretised domain
    N_c          : Specified number of intervals for Main domain
    """
    # Preallocating storage for matrix
    number_of_boxes = N_c**2
    A = np.zeros((number_of_boxes, number_of_boxes))
    J_vector_index = 0
    A_row_index = 0
    
    for i in range(number_of_boxes):
        # Calculating the adjacent boxes for box i and determining how many there are
        adj_boxes = find_surrounding_boxes(mesh_numbers, N_c, i+1)
        number_of_adj_boxes = np.size(adj_boxes)
        # Finding the total number of times a track entered that box
        J_in_value = np.sum(J_in[J_vector_index:(J_vector_index+number_of_adj_boxes)])
        # If it's 0 then we just assign that entry 0 since we can't divide by 0
        if J_in_value == 0:
            A[i, adj_boxes-1] = 0
        else:
            for j in range(number_of_adj_boxes):
                # Calculating the probability that a track moves from box i to box j
                A[i, adj_boxes[j]-1] = J_out[J_vector_index+j]/J_in_value
        
        # Moving up the vector as we go through each box
        J_vector_index += number_of_adj_boxes

    # Checking row sums are <=1
    row_sums = np.sum(A,axis=1).tolist()
    print(row_sums)
    # Writing to text file, currently gets overwritten at each run
    results_dir = "discretised_langevin_results/"
    np.savetxt(results_dir + "probability_matrix_A.txt", A, delimiter=',')
    
    return

def calculate_importance_function():
    """
    calculate_importance function computes the eigenproblem A*h = lambda*h, where lambda is the leading
    eigenvalue and h is a unique eigenvector of lambda. We take h to be the importance function of our
    probability matrix
    """
    results_dir = "discretised_langevin_results/"
    # Loading probability matrix of our discretised domain
    prob_matrix = np.loadtxt(results_dir + "probability_matrix_A.txt", delimiter=',')
    # Finding the e-values and e-vectors for the probability matrix
    evalues, evectors = np.linalg.eig(prob_matrix)
    
    # Finding in which position the leading e-value is
    leading_index = np.argmax(np.abs(evalues))
    
    # Obtaining leading e-value and it's e-vector
    leading_evalue = np.real(evalues[leading_index])
    h = np.real(evectors[:,leading_index])
    # Normalising h
    h = np.abs(h/np.max(np.abs(h)))
    
    # Writing to text file, currently gets overwritten at each run
    np.savetxt(results_dir + "importance_function_h.txt", h, delimiter=',')
    
    return h, leading_evalue

def split_or_kill_particle(weight_ratio):
    """
    split_or_kill_particle calculates whether the particle is killed or splits when it moves from one box to 
    another in the mesh. This is done using weight values taken from the importance function h.
    
    Arguments:
    weight_ratio: The weight ratio when a track changes box, h(B_(t_k))/h(B_(t_(k-1)))
    """
    # 0 if attempt to kill fails
    number_of_particles = 0
    # Splitting the particle
    if weight_ratio < 1:
        one_over_weight = 1/weight_ratio
        n = np.floor(one_over_weight).astype(int)
        p = one_over_weight - n
        bernoulli_trial = np.random.default_rng().binomial(1, p)
        if bernoulli_trial == 1:
            number_of_particles = n+1
        else:
            number_of_particles = n
    # Killing the particle
    elif weight_ratio > 1:
        p = 1 - (1/weight_ratio)
        bernoulli_trial = np.random.default_rng().binomial(1,p)
        if bernoulli_trial == 1:
            number_of_particles = -1
    return number_of_particles   

def simulate_weighted_process(n_steps, starting_point, t, T, eigval, starting_angle = np.random.default_rng().uniform(0, 2*np.pi), sigma = 1, wt = 1.):
    """
    simulate_weighted_process simulates a weighted process using the importance function h. A particle
    can split or be killed if the weight difference from step i to i+1 meets certain conditions. This provides
    a bias to the domain but tries to prevent a change in expectation.
    
    Arguments:
    n_steps        : Number of steps taken over the process
    starting_point : Starting point of track
    t              : Initial time of track
    T              : Final time of track
    eigval         : Leading eigenvalue corresponding to eigenvector h
    starting_angle : Initial angle for track, if unspecified then a random angle is chosen
    sigma          : Scaling parameter sigma to improve behaviour of model
    wt             : Initial weight of particle
    """
    results_dir = "discretised_langevin_results/"
    # Loading in importance function used to calculate the weights W_n
    h = np.loadtxt(results_dir + "importance_function_h.txt", delimiter=',')
    # Need to know the value of N_c, the number of intervals in Main Domain
    N_c = np.sqrt(np.size(h)).astype(int)
    intervals = np.linspace(0,1,N_c+1)
    # Calculating step length
    step_length = (T-t)/n_steps
    # Allocating storage for box numbers, weights and tracks
    tracks = np.zeros((1,2,n_steps+1))
    track_box_numbers = np.zeros((1,n_steps+1)).astype(int)
    track_weights = np.zeros((1,n_steps+1))
    track_angles = np.zeros((1,n_steps+1))
    # Providing initial conditions
    tracks[0,:,0]                = starting_point
    track_box_numbers[0,0], _, _ = find_box(intervals, starting_point, N_c)
    track_weights[0,0]           = wt
    track_angles[0,0]            = starting_angle
    
    # Now looping for all tracks, adding/removing particles at each step.
    for track_num in range(n_steps):
        particle_num = 0
        current_particle_number = np.size(track_box_numbers[:,track_num]) - 1
        while particle_num <= current_particle_number and current_particle_number > -1:
            # Resetting number of particles in this loop
            number_of_particles_added = 0
            # Performing a Langevin Brownian Motion for one step, not using Brownian class as that is optimised for multiple steps
            track_angles[particle_num,track_num+1]  = np.random.default_rng().normal(0, np.sqrt(step_length)*sigma) \
                + track_angles[particle_num,track_num] 
            omega_value                        = np.array([np.cos(track_angles[particle_num,track_num+1]),
                                                           np.sin(track_angles[particle_num,track_num+1])])
            tracks[particle_num,:,track_num+1] = tracks[particle_num,:,track_num] + step_length*omega_value
            
            # Calculating what box number the track has moved to
            if np.any(tracks[particle_num,:,track_num+1] > 1) or np.any(tracks[particle_num,:,track_num+1] < 0):
                tracks               = np.delete(tracks, particle_num, axis=0)
                track_box_numbers    = np.delete(track_box_numbers, particle_num, axis=0)
                track_weights        = np.delete(track_weights, particle_num, axis=0)
                track_angles         = np.delete(track_angles, particle_num, axis=0)
                current_particle_number += -1
                continue
            # Calculating box number at this point in track
            track_box_numbers[particle_num, track_num+1], _, _ = find_box(intervals, tracks[particle_num,:,track_num+1], N_c)
            # If they are in the same box, then don't bother checking weight ratio as it will be 1
            if track_box_numbers[particle_num, track_num+1] == track_box_numbers[particle_num, track_num]:
                track_weights[particle_num,track_num+1] = track_weights[particle_num,track_num]
                particle_num += 1
                continue
            # Calculating weight ratio
            track_weights[particle_num,track_num+1] =  (h[track_box_numbers[particle_num,track_num]-1]/h[track_box_numbers[particle_num,track_num+1]-1])

            # Working out number of particles in next step
            added_particles = split_or_kill_particle((track_weights[particle_num,track_num+1]*eigval))
            
            # Calculating actual weight of particle
            track_weights[particle_num,track_num+1] = track_weights[particle_num,track_num+1]*track_weights[particle_num,track_num]*eigval
            if added_particles == -1:
                # Killing particle
                tracks            = np.delete(tracks, particle_num, axis=0)
                track_box_numbers = np.delete(track_box_numbers, particle_num, axis=0)
                track_weights     = np.delete(track_weights, particle_num, axis=0)
                track_angles         = np.delete(track_angles, particle_num, axis=0)
                current_particle_number += -1
            elif added_particles > 0:
                # Adding new particles that split from main particle
                tracks               = np.append(tracks, np.repeat(np.array([tracks[particle_num,:,:]]), added_particles, axis=0), axis=0)
                track_box_numbers    = np.append(track_box_numbers, np.repeat(np.array([track_box_numbers[particle_num,:]]), added_particles, axis=0), axis=0)
                track_weights        = np.append(track_weights, np.repeat(np.array([track_weights[particle_num,:]]), added_particles, axis=0), axis=0)
                track_angles        = np.append(track_angles, np.repeat(np.array([track_angles[particle_num,:]]), added_particles, axis=0), axis=0)
                particle_num += 1
            else:
                particle_num += 1
    return tracks, track_weights, track_angles

def particle_distribution(n_samples, number_of_steps, starting_point, initial_time, final_time, segments, eval, sig = 1):
    """
    particle_distribution calculates the number of surviving particles at intervals between the initial time
    and the final time. This is then output in a histogram, one for the standard process and one for the weighted
    process
    
    Arguments:
    n_samples       : Number of sample tracks used to calculate surviving particles
    number_of_steps : Number of steps per track
    starting_point  : Starting point of each track
    initial_time    : Initial time of each track
    final_time      : Final time of each track
    segments        : Number of segments where we calculate how many particles survived at a certain point in time
    eval            : Leading eigenvalue corresponding to eigenvector h
    sig             : Scaling parameter sigma to improve behaviour of model
    """
    results_dir = "discretised_langevin_results/"
    ## Firstly simulating for the normal case
    # Calculating intervals for histogram
    interval_length = (final_time-initial_time)/segments
    time_intervals = np.around(np.linspace(initial_time,final_time,segments+1),decimals=1)
    # Preallocating array
    main_tracks = np.zeros((n_samples, 2, number_of_steps+1))
    # Initialising the Brownian class for some starting point
    b = Brownian(x0 = starting_point)
    
    # Calling Langevin Brownian Motion function
    for sample_num in range(n_samples):
        main_tracks[sample_num,:,:], _, _ = b.langevin_motion(n_steps = number_of_steps, t = initial_time, T = final_time, sigma = sig)
    
    # Calculating which step each interval ends at
    interval_of_steps = int(number_of_steps/segments)
    # Initialising values for the surviving particles at each segment
    surviving_particles = np.zeros(segments+1)
    surviving_particles[0] = n_samples
    
    # Finding which particles survive at each interval
    for i in range(segments):
        surviving_tracks = main_tracks
        number_deleted = 0
        lower_seg = i*interval_of_steps
        upper_seg = (i+1)*interval_of_steps
        for sample_num in range(np.size(main_tracks, axis=0)):
        
            if np.any(main_tracks[sample_num,:,lower_seg:upper_seg] > 1) or \
            np.any(main_tracks[sample_num,:,lower_seg:upper_seg] < 0):
                surviving_tracks = np.delete(surviving_tracks, sample_num - number_deleted, axis=0)
                number_deleted += 1
        surviving_particles[i+1] = np.size(surviving_tracks,axis=0)
        main_tracks = surviving_tracks
    
    ## Simulating for weighted process
    # Preallocating array
    weighted_tracks = np.zeros((n_samples,2,number_of_steps+1))
    
    # Loading in importance function used to calculate the weights W_n
    h = np.loadtxt(results_dir + "importance_function_h.txt", delimiter=',')
    # Need to know the value of N_c, the number of intervals in Main Domain
    N_c = np.sqrt(np.size(h)).astype(int)
    intervals = np.linspace(0,1,N_c+1)
    
    # Calculating starting weight
    starting_box,_,_ = find_box(intervals, starting_point, N_c)
    starting_weight = h[starting_box-1]
    
    # Initialising values for the surviving particles at each segment
    weighted_surviving_particles = np.zeros(segments+1)
    weighted_surviving_particles[0] = n_samples*starting_weight
    
    # Create an array of fixed size to hold all the initial points of the langevin process.
    # Each entry will be a list of lists, containing for each particle the
    # intial position, the initial angle and the initial weight
    particle_list = np.empty((segments+1,),dtype=object)
    for i in range(segments+1):
        particle_list[i] = []
    
    for sample_num in range(n_samples):
        particle_list[0].append([starting_point,0.,starting_weight])
    
    for i in tqdm(range(segments)):
        temp = particle_list[i].copy()
        new_particle_list = []
        while len(temp) > 0:
            part = temp.pop()
            weighted_track, particle_weights, particle_angles = simulate_weighted_process(interval_of_steps, part[0], \
                time_intervals[i], time_intervals[i+1], eval, starting_angle = part[1], sigma = sig, wt = part[2])
            new_part_num = np.size(weighted_track, axis=0)
            temp_wt = 0
            for j in range(new_part_num):
                new_particle_list.append([weighted_track[j,:,-1],particle_angles[j,-1],particle_weights[j,-1]])
                weighted_surviving_particles[i+1] += particle_weights[j,-1]
        particle_list[i+1] = new_particle_list.copy()
    
    # Plotting histograms for standard and weighted process
    plt.subplot(1,2,1)
    plt.bar(time_intervals + interval_length/2, surviving_particles, edgecolor="black", color="red", width = interval_length)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of surviving particles")
    
    plt.subplot(1,2,2)
    plt.bar(time_intervals + interval_length/2, weighted_surviving_particles, edgecolor="black", color="blue", width = interval_length)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of surviving weighted particles")
    
    plt.tight_layout()
    plt.savefig(results_dir + "particle_simulation.png")
    plt.show()

# Running the "script" through here for now
# Number of intervals along x and y axis in main domain
main_N_c   = 5
# Number of intervals in angular domain, there was no time to implement this further
angular_N_c = 6
# Number of tracks used to calculate Prob Matrix
no_of_tracks = 10000
# Number of steps at each track
no_of_steps = 1000

recalculate_prob_matrix = False

if recalculate_prob_matrix:
    # Finding all box numbers for both domains
    main_nums, angular_nums = calculate_mesh(angular_N_c, main_N_c)
    ## Running the Langevin tracks
    main, angular = langevin_motion_sims(no_of_tracks, no_of_steps, 0, 5, sig = 1)
    ## Calculating the J_in and J_out vectors
    main_J_in, main_J_out = tracking_main_domain(main_nums, main, main_N_c)
    # Calculating the probability matrix, saves to text file
    compute_probability_matrix(main_J_in, main_J_out, main_nums, main_N_c)

# Calculating the leading e-value and it's corresponding e-vector, loads probability matrix text file
# h function is saved to text file
h, evalue = calculate_importance_function()
print("h:", h)
print("eigenvalue: ", evalue)
# Simulating the weighted process, loads h function from text file
weight_tracks, _, _ = simulate_weighted_process(200, [0.5,0.5], 0, 1, evalue)
print(weight_tracks)
particle_distribution(100, 200, [0.5,0.5], 0, 3, 10, evalue, sig = 2)



