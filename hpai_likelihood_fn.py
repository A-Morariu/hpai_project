import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
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
        location_data (Tensor - float64): 2-D array of the location coordinates of (all) units in the population 

    Returns:
        Tensor - float64: fn which returns the spatial kernel 
    """
    # compute distances between farms (i.e. [rho(i,j)] )
    farm_distance_matrix = pairwise_distance(location_data)
    farm_distance_matrix = tf.convert_to_tensor(
        farm_distance_matrix, dtype=DTYPE)

    def square_exponential_kernel(parameters):
        """Square exponential function - parameter phi places a loose cut off at phi units away from data point i 

        Mathematically: k_{SE}(i,j) = exp(- ([rho(i,j)] / phi **2)

        Args:
            parameters (float64): numeric value for cut off in spatial process

        Returns:
            Tensor - float64: square matrix of the pairwise spatial pressure applied by unit j (colm) onto unit i (row)
        """
        parameters = tf.convert_to_tensor(parameters, DTYPE)
        partial_step = tf.math.multiply(farm_distance_matrix, 1/parameters)
        return tf.math.exp(-tf.math.square(partial_step))

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
        tensor: len(infection_times) x len(infection_times) tensor of exposure
        durations 
    """

    return (
        tf.math.minimum(infection_times[:, tf.newaxis],
                        removal_times[tf.newaxis, :]) -
        tf.math.minimum(infection_times[:, tf.newaxis],
                        infection_times[tf.newaxis, :])
    )

##########################
# Regression Component
##########################


def generate_regression_pressure(characteristic_data=None):
    """Instantiate fn for regression component of model

    Args:
        farm_characteristics_data (_type_, optional): Factor variables for each farm unit. Defaults to None.

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
