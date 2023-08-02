"""Module providing access to creating namedtuple structures"""
import collections

import numpy as np
import pandas as pd
import scipy as sp

import tensorflow as tf
import tensorflow_probability as tfp

# Preamble

# Aliasing
DTYPE = tf.float64

tfd = tfp.distributions

tf_map = tf.nest.map_structure

# User defined types
ParameterTuple = collections.namedtuple(
    'ParameterTuple', 'regression spatial removal')

RWMHResult = collections.namedtuple(
    'RWMHResult', 'is_accepted current_state current_state_log_prob')

# Import data - To do
farm_locations = pd.read_csv('data_files/farm_locations.csv')

# Compute likelihood

# Helper function


def pairwise_distance(location_data):
    """Compute pairwise distance matrix between farms

    Args:
        farm_locations_data (DataFrame): Lat-Long coordinates of farms

    Returns:
        tensor: tensor of Euclidean distances between entities
                Dim = len(farm_location_data)
    """
    return tf.convert_to_tensor(
        sp.spatial.distance.squareform(
            sp.spatial.distance.pdist(location_data)
        ),
        dtype=DTYPE
    )

##########################
# Spatial kernel
##########################


def generate_spatial_kernel(location_data):
    """Closure which returns a specific spatial kernel

    eg. square exponential, euclidean, etc

    Args:
        location_data (Tensor - float64): 2-D array of the location 
        coordinates of (all) units in the population 

    Returns:
        Tensor - float64: fn which returns the spatial kernel 
    """
    # compute distances between farms (i.e. [rho(i,j)] )
    farm_distance_matrix = pairwise_distance(location_data)
    farm_distance_matrix = tf.convert_to_tensor(
        farm_distance_matrix, dtype=DTYPE)

    def square_exponential_kernel(parameters):
        """Square exponential function - parameter phi places a loose
        cut off at phi units away from data point i 

        Mathematically: k_{SE}(i,j) = exp(- ([rho(i,j)] / phi **2)

        Args:
            parameters (float64): numeric value for cut off in spatial process

        Returns:
            Tensor - float64: square matrix of the pairwise spatial pressure 
            applied by unit j (colm) onto unit i (row)
        """
        parameters = tf.convert_to_tensor(1/parameters, DTYPE)
        partial_step = tf.math.multiply(parameters, farm_distance_matrix)
        return tf.math.exp(-1 * tf.math.square(partial_step))

    return square_exponential_kernel


def generate_waifw(infection_times, removal_times):
    """
    Compute a WAIFW (who acquired infection from who) matrix 
    given tensors of infection and removal times. 

    Args:
        infection_times (datetime): array of infection times 
        removal_times (datetime): array of removal times 

    Returns:
        tensor: len(infection_times) x len(infection_times) tensor of 1s and 0s
    """

    infection_times = tf.convert_to_tensor(infection_times)
    removal_times = tf.convert_to_tensor(removal_times)

    # use the expand_dim to do the [I,:] trick
    waifw = tf.math.logical_and(
        # compare infections to infections: I_i <= I_j
        infection_times[tf.newaxis, :] < infection_times[:, tf.newaxis],
        # compare infections to removals: I_j <= R_i
        infection_times[:, tf.newaxis] < removal_times[tf.newaxis, :]
    )
    return tf.cast(waifw,
                   dtype=DTYPE)


def generate_exposure(infection_times, removal_times):
    """
    Compute exposure matrix given two tensors of infections and removal times

    Args:
        infection_times (datetime): array of infection times 
        removal_times (datetime): array of removal times 

    Returns:
        Tensor - float64: len(infection_times) x len(infection_times) tensor of exposure
        durations 
    """
    infection_times = tf.convert_to_tensor(infection_times, dtype=DTYPE)
    removal_times = tf.convert_to_tensor(removal_times, dtype=DTYPE)

    return (
        tf.math.minimum(infection_times[:, tf.newaxis],
                        removal_times[tf.newaxis, :]) -
        tf.math.minimum(infection_times[:, tf.newaxis],
                        infection_times[tf.newaxis, :])
    )


def generate_infectious_duration(infection_times, removal_times):
    """Compute the infectious duration of all infective units

    Args:
        infection_times (datetime): vector of infection times
        removal_times (datetime): vector of removal times

    Returns:
        Tensor-float64: 1D tensor of infectious duration (period which 
        an individual exerts infectious pressyre on other units in the system)
    """
    infection_times = tf.convert_to_tensor(infection_times, dtype=DTYPE)
    removal_times = tf.convert_to_tensor(removal_times, dtype=DTYPE)
    return tf.math.subtract(x=removal_times,
                            y=infection_times)

##########################
# Regression Component
##########################


def generate_regression_pressure(characteristic_data=None):
    """Instantiate fn for regression component of model

    Args:
        farm_characteristics_data (_type_, optional): Factor variables 
        for each farm unit. Defaults to None.

    Returns:
        fn: exp(alpha + beta * data)
    """
    characteristic_data = tf.convert_to_tensor(characteristic_data, DTYPE)

    def compute_regression(parameters):
        parameters = tf.convert_to_tensor(parameters, DTYPE)
        # note: char_data is a column vector so this needs to change for general case
        one_padded_data = tf.stack([np.full(np.shape(characteristic_data), 1.0),
                                    characteristic_data], axis=-1)
        regression = tf.matmul(one_padded_data, tf.expand_dims(parameters, -1))
        expontiated_regression = tf.math.exp(-regression)
        return expontiated_regression

    return compute_regression

##########################
# Infectious pressure
##########################


def generate_pairwise_hazard_fn(location_data, characteristic_data):
    """_summary_

    Args:
        farm_characteristics_data (_type_): features of farms, including a 1s column for regression
        farm_locations_data (_type_): Northing-Easting coordinates of farms

    Returns:
        fn: fn which outputs a tensor of pairwise hazard rates
    """
    spatial_kernel = generate_spatial_kernel(location_data)
    regression_kernel = generate_regression_pressure(characteristic_data)

    def compute_hazard(parameters_tuple):

        # spatial component - already exponentiated!
        spatial = spatial_kernel(parameters_tuple.spatial)

        # regression component - already exponentiated!
        regression = regression_kernel(parameters_tuple.regression)
        return tf.math.multiply(spatial,
                                regression
                                )

    return compute_hazard

############
# Removal fn
############


def generate_removal_fn(infection_time, removal_time):
    """Instantiate fn governing the removal process. 
    Close over computing the time difference (D)

    Args:
        infection_time (datetime): _description_
        removal_time (datetime): _description_

    Returns:
        fn: fn for the waiting time to the removal event
    """
    time_diff = removal_time - infection_time

    def removal_fn(parameters):
        """Exponential waiting time 
        r_j - i_j ~ Exp(gamma)

        Args:
            parameters (float64): rate parameter of an exponential distribution 

        Returns:
            float: log_prob of the differences
        """
        return tfp.distributions.Exponential(
            rate=parameters).log_prob(time_diff)
    return removal_fn

##########################
# Likelihood
##########################

# block 1 - event rate
# math: [log (WAIFW cdot (H_{I,I}) cdot 1v )] cdot 1v


def event_rate_block(waifw_matrix, hazard_matrix):
    """Compute event rate portion of log-likelihood for an SIR model

    Args:
        waifw_matrix (float64): Direction of infectious pressure matrix
        hazard_matrix (float64): Pairwise hazard rate fn evaluation

    Returns:
        float64: Cumulative event rate in the sytem
    """
    log_rates = tf.math.log(
        tf.einsum('ij -> i', tf.matmul(a=waifw_matrix, b=hazard_matrix)))
    log_rates = tf.where(
        tf.math.is_inf(log_rates),
        tf.constant(0.0, dtype=DTYPE),
        log_rates)  # remove np.inf (initial inf and occults)
    return tf.reduce_sum(log_rates)

# This function hides the matrix multiplications within the
# log_ll computation - no need to write as a closure

# block 2 - infectious pressure p.w. approximation
# math: ((E cdot H_{I,I} ) cdot 1v cdot 1v


def infectious_pressure_block(hazard_matrix, exposure_matrix):
    """_summary_

    Args:
        hazard_matrix (_type_): _description_
        exposure_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    infection_pressure = tf.matmul(a=hazard_matrix, b=exposure_matrix)
    return tf.reduce_sum(infection_pressure)

# function hides the matrix multiplications within the
# log_ll computation - no need to write as a closure

# block 3 - removal process
# math: log_prob(D)


def removal_process_block(known_event_times):
    """Closure over the known event times to evaluate the removal 
    process log prob

    Args:
        known_event_times (datetime): array of known event times in 
        an epidemic process 

    Returns:
        fn: fn which evaluates the log prob of the removal process 
        given a rate
    """

    # Unknown event times can be moved one level deeper into the
    # closure if they are unknown (i.e. are model parameters)

    infection_time = known_event_times['infection_time']
    removal_time = known_event_times['removal_time']

    def removal_process_block_eval(removal_rate):
        # TO DO: add parameter to create a removal rate fn?

        # set exponential removal times with rate = removal_rate
        removal_process = tfp.distributions.Exponential(rate=removal_rate)

        # compute D
        infectious_duration = removal_time - infection_time

        # compute log prob - account for P(X = 0) = lambda in exp distribution
        # handles occults since their infectious duration is 0 so they don't
        # have a removal time - convention is that occults had I = R = T_maxlog
        log_prob_duration = tf.where(infectious_duration > 0,
                                     removal_process.log_prob(
                                         infectious_duration),
                                     0.0)

        return tf.reduce_sum(log_prob_duration)
    return removal_process_block_eval


def log_ll(hazard_rate_function, known_event_times):
    """Closure over the data for the likelihood function
    of the model

    Args:
        hazard_rate_function (fn): pairwise hazard rate fn (closes 
        over the same data as the log_ll)
        known_event_times (datetime): array of known epidemic 
        event times

    Returns:
        fn: fn which evaluates the log-likelihood of an epidemic model
    """
    @tf.function(jit_compile=False)
    def log_ll_eval(parameters_tuple):
        # Compute matrices
        hazard_mat = hazard_rate_function(parameters_tuple)

        waifw_mat = generate_waifw(
            infection_times=known_event_times
            ['infection_time'],
            removal_times=known_event_times
            ['removal_time'])

        exposure_mat = generate_exposure(
            infection_times=known_event_times['infection_time'],
            removal_times=known_event_times['removal_time'])
        # Block 1: [log (WAIFW cdot (H_{I,I}) cdot 1v )] cdot 1v

        block1 = event_rate_block(
            waifw_matrix=waifw_mat, hazard_matrix=hazard_mat)

        # Block 2: - ((E cdot H_{I,I} ) cdot 1v cdot 1v

        block2 = infectious_pressure_block(hazard_matrix=hazard_mat,
                                           exposure_matrix=exposure_mat)

        # Block 3: log_prob(D)

        removal_process = removal_process_block(
            known_event_times=known_event_times)

        # evaluate the contribution
        block3 = removal_process(removal_rate=parameters_tuple.removal)

        return block1 + block2 + block3
    return log_ll_eval

##########################
# Priors
##########################


def prior_distributions_block(initial_values_tuple):
    """Instantiate non-informative priors for each block of inference

    Returns:

    Args:
        initial_values_tuple (ParameterTuple): namedtuple of initial 
        parameter values. The shapes of these vectors are used to set 
        up the structure for further prior distributiuon evaluations

    Returns:
        fn: fn with input current parameter values to evaluate the 
        prior distribution
    """

    # Goal: priors to be initialized w/ arguments
    # (initial_values, distribution, distribution_param)
    # inital value are used to set up the shapes of the priors

    # initialize tfp.distributions
    reg_priors = tfp.distributions.Normal(loc=tf.broadcast_to(tf.constant(
        0., dtype=DTYPE), tf.shape(initial_values_tuple.regression)), scale=10.)

    spatial_priors = tfp.distributions.Normal(loc=tf.broadcast_to(tf.constant(
        0., dtype=DTYPE), tf.shape(initial_values_tuple.spatial)), scale=10.)

    removal_priors = tfp.distributions.Normal(loc=tf.broadcast_to(tf.constant(
        0., dtype=DTYPE), tf.shape(initial_values_tuple.removal)), scale=10.)

    @tf.function(jit_compile=True)
    def prior_eval(parameters):
        """Evaluation of the log-prob of prior distributions at specific parameter values

        Args:
            parameters (ParameterTuple): current parameter values in an inference algorithm

        Returns:
            float64: sum of the log-probs of the evaluated priors
        """
        reg_block = tf.reduce_sum(reg_priors.log_prob(
            parameters.regression))
        spatial_block = tf.reduce_sum(
            spatial_priors.log_prob(parameters.spatial))
        removal_block = tf.reduce_sum(
            removal_priors.log_prob(parameters.removal))

        return reg_block + spatial_block + removal_block
    return prior_eval

##########################
# Target log prob
##########################


def target_log_prob_fn(log_likelihood_fn, prior_dist_fn):
    """Combine log-likelihood and prior distributions to create the 
    target log prob fn used in inference

    Args:
        log_likelihood_fn (fn): log-likelihood fn of the model
        prior_dist_fn (fn): prior distribution fn of the model parameters

    Returns:
        float64: evaluation of the target log prob
    """
    # initialize the two parts
    def target_log_prob_eval(parameters_tuple):
        return log_likelihood_fn(parameters_tuple) + prior_dist_fn(parameters_tuple)
    return target_log_prob_eval
