"""Module providing access to creating namedtuple structures"""
import collections

import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
import tensorflow_probability as tfp

# Preamble

# Aliasing
DTYPE = tf.float64

tfd = tfp.distributions

tf_map = tf.nest.map_structure

# User defined types
ParameterTuple = collections.namedtuple(
    'ParameterTuple', 'mean')

RWMHResult = collections.namedtuple(
    'RWMHResult', 'is_accepted current_state current_state_log_prob')

# Data creation - toy example

toy_param = {'loc': tf.constant(3.5, dtype=DTYPE),
             'scale': tf.constant(1.34, dtype=DTYPE)
             }

toy_gaussian = tfp.distributions.Normal(
    loc=toy_param['loc'],
    scale=toy_param['scale']).sample(
    100, seed=20060420)


def toy_log_ll(data):
    """log-likelihood for a normal distribution

    Args:
        data (float64): sample of a noraml distribution
    """
    def target_log_prob(parameters):
        # prior distribution
        prior_dist = tfp.distributions.Normal(
            loc=tf.constant(0., DTYPE),
            scale=tf.constant(10., DTYPE)).log_prob(parameters)

        # likelihood
        log_ll_eval = tfp.distributions.Normal(
            loc=parameters, scale=toy_param['scale']).log_prob(data)
        return tf.reduce_sum(log_ll_eval) + tf.reduce_sum(prior_dist)

    return target_log_prob


def generate_gaussian_proposal(scale):
    """Instantiate a Gaussian proposal for an MCMC kernel 

    Args:
        scale (float64): the scale of the proposal distribution. 
        Must contain only positive values. 

    Returns:
        fn: proposal function with a fixed scale parameter. The
        function takes in the current chain state and a seed, 
        and returns a sample of size 1 matching the current 
        state dimensions
    """

    def gaussian_proposal(current_state, seed):
        """Propose next state in an MCMC chain given a current
        state and a seed 

        Args:
            current_state (float64): vector of values of the 
            current chain state
            seed (int): PRNG seed

        Returns:
            float64: suggested next state for the chain
        """
        loc = current_state
        isotropic_normal = tfp.distributions.Normal(
            loc=loc, scale=scale)
        return isotropic_normal.sample(seed=seed)
    return gaussian_proposal

# inference algo


def random_walk_metropolis_hastings_kernel(
        target_log_prob_fn, initial_state, proposal_scale):
    """Runs one step of the Metropolis-Hastings algorithm.
    Proposal distribution is symmetric meaning there is
    no need for any adjustments to the log-acceptance ratio

    Args:
        target_log_prob_fn (fn): a *partial* evaluation of the
        model target log prob function which moves *only* the
        parameters of interest of the inference block 
        proposal_scale (float64): vector of the proposal
        distribution scales for the parameters of interest
        of the inference block

    Returns:
        current_state (float64): state of the chain after the step
        has been run
        kernel_result (RWMHResult): tracing tuple of information
        from the algorithm

    """
    proposal_distribution = generate_gaussian_proposal(scale=proposal_scale)

    def bootstrap_result(current_state):
        """Move this inside the kernel 

        Args:
            initial_state_tuple (_type_): _description_
            target_log_prob_fn (_type_): _description_

        Returns:
            _type_: _description_
        """
        kernel_output = RWMHResult(
            is_accepted=tf.ones_like(
                target_log_prob_fn(current_state),
                dtype=tf.bool),
            current_state=current_state,
            current_state_log_prob=target_log_prob_fn(current_state))
        return ParameterTuple(kernel_output)

    @tf.function(jit_compile=True)
    def random_walk_metropolis_hastings_one_step(
            current_state, previous_kernel_result, seed):
        """_summary_

        Args:
            current_state (ParameterTuple): _description_
            previous_kernel_result (RWMHResult): _description_
            seed (int): PRNG seed

        Returns:
            _type_: _description_
        """
        current_state_log_prob = previous_kernel_result[-1].current_state_log_prob
        # Step 1: Seed handling - Can we do this outside?
        proposal_seed, accept_seed = tfp.random.split_seed(
            seed, n=2, salt="RWMHKernel")

        # Step 2: Propose next step
        next_state = proposal_distribution(current_state=current_state,
                                           seed=proposal_seed)

        # Step 3: Evaluate the target-log-prob of the proposal
        # IMPORTANT: inputted target_log_prob_fn is a partial evaluation
        # of the model target_log_prob in the main function

        next_target_log_prob = target_log_prob_fn(next_state)

        # Step 4: Compute log-accept ratio
        log_accept_ratio = next_target_log_prob - \
            current_state_log_prob

        # Step 5: Check accept/reject
        log_uniform = tf.math.log(tfp.distributions.Uniform(
            high=tf.constant(1., dtype=DTYPE)).sample(seed=accept_seed))

        is_accepted = log_uniform < log_accept_ratio

        # Step 6: Update the current_state and kernel_results
        current_state = tf.cond(
            is_accepted, lambda: next_state, lambda: current_state)
        current_state_log_prob = tf.cond(
            is_accepted, lambda: next_target_log_prob,
            lambda: current_state_log_prob)

        kernel_results = RWMHResult(
            is_accepted=is_accepted,
            current_state=current_state,
            current_state_log_prob=current_state_log_prob
        )
        return current_state, ParameterTuple(kernel_results)

    return bootstrap_result(initial_state), random_walk_metropolis_hastings_one_step


def sample_chain(
        current_state, previous_kernel_results, kernel, seed, num_results=15):
    """Implements Markov chain Monte Carlo via repeated `TransitionKernel` steps

    This acts as a generic loop wrapper which takes in a kernel and reapplies it
    to the current state. The previous state is an instance of the bootstrap
    result and is used to cache computations done in the previous iteration 
    of the algorithm. 

    Args:
        current_state (ParameterTuple): `Tensor` or Python `list` of 
        `Tensor`s representing the current state(s) of the Markov chain(s).
        previous_kernel_results (RWMHResults): A `Tensor` or a nested 
        collection of `Tensor`s representing internal calculations made
        within the previous call to this function (or as 
        returned by `bootstrap_results`) 
        kernel (fn): an *instance* of a transition kernel which implements 
        a one step fn 
        num_results (int, optional): number of iterations to run the MCMC 
        algorithm for. Defaults to 15.

    Returns:
        parameters_samples: A nest collection of 'Tensor's containing the
        values of the MCMC chain
        kernel_trace: A nested collection of 'Tensor's containing cached
        calculations of the MCMC algorithm for each iteration (used in 
        diagnostics and tuning)
    """
    global kernel_trace
    # Create MCMC tracers
    parameter_samples = tf_map(
        lambda x: tf.TensorArray(dtype=x.dtype, size=num_results),
        current_state)

    kernel_trace = tf_map(
        lambda x: tf.TensorArray(dtype=x.dtype, size=num_results),
        previous_kernel_results)

    def cond(iterator,
             _2,
             _3,
             _4,
             _5,
             _6):
        """_summary_

        Args:
            iterator (_type_): _description_

        Returns:
            _type_: _description_
        """
        return iterator < num_results

    def body(iterator,
             current_state,
             previous_kernel_result,
             parameter_samples,
             kernel_trace,
             seed):
        """_summary_

        Args:
            iterator (_type_): _description_
            current_state (_type_): _description_
            previous_kernel_result (_type_): _description_
            parameter_samples (_type_): _description_
            mcmc_results (_type_): _description_
            seed (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Seed handling
        this_seed, next_seed = tfp.random.split_seed(seed, n=2, salt="body")

        # Perform one step of the chain
        next_state, next_kernel_result = kernel(
            current_state=current_state,
            previous_kernel_result=previous_kernel_result,
            seed=this_seed
        )

        parameter_samples = tf_map(
            lambda x, a: a.write(iterator, x),
            next_state, parameter_samples)

        # Track the outcome - KERNEL STATE(S)
        kernel_trace = tf_map(lambda x, a: a.write(iterator, x),
                              next_kernel_result, kernel_trace)

        return (iterator + 1,
                next_state,
                next_kernel_result,
                parameter_samples,
                kernel_trace,
                next_seed)

    # Run chain(s)
    (_1,
     _2,
     _3,
     parameter_samples,
     kernel_trace,
     _4) = tf.while_loop(cond=cond,
                         body=body,
                         loop_vars=(0,
                                    current_state,
                                    previous_kernel_results,
                                    parameter_samples,
                                    kernel_trace,
                                    seed))

    # formatting
    parameter_samples = tf_map(
        lambda x: x.stack(), parameter_samples)

    kernel_trace = tf_map(lambda x: x.stack(),  kernel_trace)

    return parameter_samples, kernel_trace


# Run the algorithm - should run faster than the notebook
# so this is the next to do so we can debug here

kernel_output_structure, kernel_fn = random_walk_metropolis_hastings_kernel(
    target_log_prob_fn=toy_log_ll(data=toy_gaussian),
    initial_state=tf.constant(5.0, dtype=DTYPE),
    proposal_scale=1.)

N = 10_000

chains, results = sample_chain(
    current_state=tf.constant(5.0, dtype=DTYPE),
    previous_kernel_results=kernel_output_structure,
    kernel=kernel_fn,
    seed=20220807,
    num_results=N)

# acceptance rate:
print(f'MCMC sampler acceptance rate: {100*np.sum(results[0].is_accepted)/N}%')

# Visual diagnostics - Create a figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Trace plot
axs[0].plot(chains, linestyle='-', color='b', label='Trace', linewidth=0.5)
axs[0].axhline(y=np.mean(chains), color='r',
               linestyle='--', label='Mean', linewidth=1.5)
axs[0].axhline(y=3.5, color='g', linestyle='--',
               label='True value', linewidth=1.5)
# Add labels and title
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Value')
axs[0].set_title('Trace Plot of MCMC Samples')
axs[0].legend()

# Acceptance rate plot
axs[1].plot(np.cumsum(results[0].is_accepted) / range(1, N + 1),
            linestyle='-', color='b', label='Acceptance rate', linewidth=1)
axs[1].axhline(y=0.26, color='g', linestyle='--',
               label='Optimal value', linewidth=1.5)
# Add labels and title
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Value')
axs[1].set_title('Acceptance Rate of MCMC Samples')
axs[1].legend()

# Log posterior plot
axs[2].plot(results[0].current_state_log_prob, linestyle='-',
            color='b', label='Log-Prob', linewidth=1)
# Add labels and title
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Value')
axs[2].set_title('Log-Posterior of MCMC Samples')
axs[2].legend()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
