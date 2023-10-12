"""Module providing access to creating namedtuple structures"""
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp

# Preamble
sns.set_theme()

# Aliasing
DTYPE = tf.float64

tfd = tfp.distributions

tf_map = tf.nest.map_structure

# user defined types
ParameterTuple = collections.namedtuple(
    'ParameterTuple', 'loc scale')

ChainStateTuple = collections.namedtuple(
    'ChainStateTuple', 'position log_prob seed')

McmcResult = collections.namedtuple(
    'McmcResult', 'is_accepted current_state current_state_log_prob')

# target log probility function


def generate_target_log_prob(*, data):
    """
    Returns the log-likelihood function of a normal distribution for a given sample.

    Args:
        data: A float64 Tensor representing a sample from a normal distribution.

    Returns:
        A Tensor representing the log-likelihood of the sample.

    Raises:
    ValueError: If the data is not a float64 Tensor.
    """

    def target_log_prob(*, loc, scale):
        """Evaluate log-likelihood of data from a normal distribution.



        Args:
            loc (float64): loc/mean of normal distribution
            scale (float64): scale/standard deviation of normal distribution

        Returns:
            float64: evaluation of log-likelihood of data from a normal distribution
        """
        loc = tf.cast(loc, dtype=DTYPE)
        scale = tf.cast(scale, dtype=DTYPE)

        # prior distribution
        prior_dist = tfp.distributions.Normal(
            loc=tf.constant(0., DTYPE),
            scale=tf.constant(10., DTYPE)).log_prob(loc) + \
            tfp.distributions.Exponential(rate=tf.constant(2, DTYPE)).log_prob(scale)

        # likelihood
        log_ll_eval = tfp.distributions.Normal(
            loc=loc, scale=scale).log_prob(data)
        return tf.reduce_sum(log_ll_eval) + tf.reduce_sum(prior_dist)

    return target_log_prob


def generate_partial_target_log_prob(target_log_prob_fn, **kwargs):
    """
    Reimplementation of the partial function evaluation in the functools package. 
    A target_log_prob_fn and a namedTuple of all the arguments the target_log_prob_fn
    takes. The inner closure takes a name-value pairing for the part of the 
    target_log_prob_fn which will be the subject of inference in the step. In other 
    words, it performs the Gibbs step of conditioning the target_log_prob_fn on *all* 
    model variables with the exception of one which corresponds to the accompaying
    TransitionKernel (this will perform a one_step evaluation on the target variable)

    Args:
        target_log_prob_fn: the target log-probability of the model
        **kwargs: a namedTuple of parameters and values. This is to be the current_state 
            of the chain

    Return:
        partial_target_log_prob_fn:
    """
    def partial_target_log_prob_fn(**target_kwargs):
        """
        Create an conditional probability function which takes only one arguement. Namely,
        the variable that is the scope of the current Gibbs step of the 
        Metropolis-within-Gibbs algorithm.

        Return: 
            target_log_prob_fn(**partial_kwargs, **kwargs): an evaluation of the target_log_prob_fn 
                with all variables fixed
        """
        return target_log_prob_fn(**target_kwargs, **kwargs)

    return partial_target_log_prob_fn


def get_conditional_variables(*, parameters, target_variable):
    """Identify the variables to condition on in the Gibbs step of the
    Metropolis-within-Gibbs algorithm.

    Args:
        parameters (ParameterTyple): variables for the model
        target_variable (string): current variable to be updated

    Returns:
        dict: dictionary of conditional variables and their values, these
        are the fixed elements for the Gibbs step
    """
    # k - variable name (i.e. key)
    # v - variable value (i.e. value)
    return {var_name: var_value for var_name, var_value in parameters._asdict().items() if var_name not in target_variable}
# proposal distributions


def generate_gaussian_proposal(*, proposal_scale):
    """Instantiate a Gaussian proposal for an MCMC kernel 

    Args:
        proposal_scale (float64): the scale of the proposal distribution. 
        Must contain only positive values. If a single value is provided,
        it will be used for all variables. If multiple values are provided,
        they must match the number of variables in the model 

    Returns:
        fn: proposal function with a fixed scale parameter. The
        function takes in the current chain state and a seed, 
        and returns a sample of size 1 matching the current 
        state dimensions
    """
    proposal_scale = tf.cast(proposal_scale, DTYPE)

    def gaussian_proposal_fn(*, position: tuple, seed):
        """Propose next state in an MCMC chain given a current
        state and a seed 

        Args:
            position (tuple): tuple of names and values of the 
            current chain state
            seed (int): PRNG seed

        Returns:
            float64: suggested next state for the chain
        """
        # Convert components of position to tensors if needed
        if not isinstance(position, (tuple, list)):
            position = (position,)
        target_variables = [tf.convert_to_tensor(p) for p in position]

        seeds = tfp.random.split_seed(
            seed=seed, n=len(position),
            salt="GaussianProposal")

        proposal = [
            tfp.distributions.Normal(loc=loc, scale=scale).sample(seed=s)
            for loc, scale, s in (zip(target_variables,
                                      proposal_scale if tf.size(proposal_scale) > 1 else tf.repeat(proposal_scale, len(target_variables)),
                                      seeds))
        ]
        if not isinstance(position, (tuple, list)):
            return proposal[0]

        return position.__class__(*proposal)

    def log_acceptance_correction_fn(*, position, proposed_position):
        return tf.constant(0., dtype=DTYPE)

    return gaussian_proposal_fn, log_acceptance_correction_fn


# inference kernels


def random_walk_metropolis_kernel(
        *, target_log_prob_fn, proposal_distribution,  # initial_state
):
    """Generate a random walk Metropolis kernel for a given target_log_prob_fn

    _extended_summary_

    Args:
        target_log_prob_fn (callable): target log probability function of the model
        proposal_distribution (callabe): proposal distribution for the kernel
    """
    proposal_distribution, log_acceptance_correction_fn = proposal_distribution

    def random_walk_metropolis_one_step(*, current_state):
        position, log_prob, seed = current_state._asdict().values()

        # Step 1: Seed handling
        proposal_seed, accept_seed = tfp.random.split_seed(
            seed, n=2, salt="RWMHKernel")

        # Step 2: Propose next step
        next_state = proposal_distribution(position=position,
                                           seed=proposal_seed)
        print(f'next_state: {next_state}')
        # PROBLEM: need to stitch together a name-value pairing

        # Step 3: Evaluate the target-log-prob of the proposal
        # IMPORTANT: inputted target_log_prob_fn is a partial evaluation
        # of the model target_log_prob in the main function

        next_log_prob = target_log_prob_fn(next_state)

        # Step 4: Compute log-accept ratio
        log_accept_ratio = next_log_prob - log_prob

        # Step 5: Check accept/reject
        log_uniform = tf.math.log(tfp.distributions.Uniform(
            high=tf.constant(1., dtype=DTYPE)).sample(seed=accept_seed))

        is_accepted = log_uniform < log_accept_ratio

        # Step 6: Update state
        new_state = tf.cond(
            is_accepted, lambda: next_state, lambda: position)
        new_log_prob = tf.cond(
            is_accepted, lambda: next_log_prob,
            lambda: log_prob)

        new_state = ChainStateTuple(position=new_state,
                                    log_prob=new_log_prob,
                                    seed=42)
        return new_state
    return random_walk_metropolis_one_step


def MwG_step(*, target_log_prob_fn, initial_state):
    alias_kernel = random_walk_metropolis_kernel(
        target_log_prob_fn=target_log_prob_fn,
        proposal_distribution=generate_gaussian_proposal(proposal_scale=1.))

    for variable_name, variable_value in initial_state._asdict().items():

        # get coniditional variables and values
        conditional_variables = get_conditional_variables(
            parameters=initial_state, target_variable=variable_name)

        # get partial target log prob function
        partial_target_log_prob_fn = generate_partial_target_log_prob(
            target_log_prob_fn=target_log_prob_fn, **conditional_variables)

        # evaluate the target log prob function
        print(
            f"Changing {variable_name} evaluation: {partial_target_log_prob_fn(**{variable_name: variable_value})}")

        #
        kernel = alias_kernel(
            current_state=ChainStateTuple(
                position=variable_value,
                log_prob=partial_target_log_prob_fn(
                    **{variable_name: variable_value}),
                seed=42))
        print(kernel)

    return None


# Testing
# Data creation - toy example

toy_param = ParameterTuple(**{'loc': tf.constant(3.5, dtype=DTYPE),
                              'scale': tf.constant(1.34, dtype=DTYPE)
                              })

toy_gaussian = tfp.distributions.Normal(**toy_param._asdict()).sample(
    1000, seed=20060420)

TLP_fn = generate_target_log_prob(data=toy_gaussian)

# full TLP evaluation
print(f"Full TLP evaluation: {TLP_fn(**toy_param._asdict())}")

# partial TLP evaluation
MwG_step(target_log_prob_fn=TLP_fn, initial_state=toy_param)
