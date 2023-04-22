import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

import tensorflow as tf
import tensorflow_probability as tfp

# Preamble

DTYPE = np.float32

tfd = tfp.distributions

# Import data - To do
farm_locations = pd.read_csv('data_files/farm_locations.csv')

# Compute likelihood

# Helper function


def pairwise_distance(farm_locations_data):
    """Compute pairwise distance matrix between farms

    Args:
        farm_locations_data (DataFrame): Lat-Long coordinates of farms

    Returns:
        tensor: tensor of Euclidean distances between entities
                Dim = len(farm_location_data)
    """
    return tf.convert_to_tensor(
        sp.spatial.distance.pdist(farm_locations_data),
        dtype=DTYPE
    )


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
# Spatial kernel
##########################


def generate_spatial_kernel(farm_distance_matrix):
    """Compute the square exponential spatial kernel given a pairwise distance matrix

    Args:
        farm_distance_matrix (tensor): matrix of pairwise distance

    Returns:
        function: function taking in spatial pressure parameter based on 
    """
    def square_exponential_kernel(parameters):
        return tf.math.exp(
            - tf.math.square(tf.math.divide(farm_distance_matrix, parameters)
                             ))
    return square_exponential_kernel


def test_spatial_kernel():
    """Test case for spatial kernel

    Returns:
        _type_: _description_
    """
    fake_distances = tf.constant([[0, 0.1, 1.5],
                                  [0.1, 0, 0.73],
                                  [1.5, 0.73, 0]],
                                 dtype=DTYPE,
                                 name='fake distances')
    fake_parameter = tf.constant([0.5], dtype=DTYPE, name='fake phi')

    fake_spatial_pressure_fn = generate_spatial_kernel(fake_distances)

    return fake_spatial_pressure_fn(fake_parameter)


print(f'Spatial values {test_spatial_kernel()} - correct!')

##########################
# Infectious pressure
##########################


def generate_pairwise_hazard_fn(farm_characteristics_data, farm_distance_matrix):
    """_summary_

    Args:
        farm_characteristics_data (_type_): features of farms, including a 1s column for regression
        farm_locations_data (_type_): Northing-Easting coordinates of farms

    Returns:
        fn: fn which outputs a tensor of pairwise hazard rates 
    """
    spatial_kernel = generate_spatial_kernel(farm_distance_matrix)

    def compute_hazard(parameters):
        # regression component - have a column of 1s in the data
        print(parameters[:-1])
        regression = tf.exp(tf.math.multiply(
            farm_characteristics_data, parameters[:-1])
        )

        # spatial component - alreay exponentiated!
        print(parameters[-1])
        spatial = spatial_kernel(parameters[-1])

        return regression + spatial

    return compute_hazard


def test_hazard_fn():
    """Test case for hazard fn

    Returns:
        _type_: _description_
    """
    # 3 farm population
    fake_char = tf.constant([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]],
                            dtype=DTYPE,
                            name='fake data')
    fake_distances = tf.constant([[0, 0.1, 1.5],
                                  [0.1, 0, 0.73],
                                  [1.5, 0.73, 0]],
                                 dtype=DTYPE,
                                 name='fake distances')
    fake_parameter = tf.constant(
        [3.14, 0.1, 1.2, 0.5], dtype=DTYPE, name='fake parameters')

    fake_hazard_fn = generate_pairwise_hazard_fn(
        farm_characteristics_data=fake_char, farm_distance_matrix=fake_distances)

    return fake_hazard_fn(fake_parameter)


print(f'Hazard values {test_hazard_fn()} - !')
##########################
