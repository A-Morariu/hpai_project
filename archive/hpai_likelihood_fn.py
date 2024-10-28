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

# Section 1 - Likelihood specification
# This section is dedicated to outlining the likelihood function
# of the epidemic model we are using. All transitions and events
# are highlighted in the mathematical equation. From an implementation
# perspective, we take a functional programming approach to allocate
# intermediary tasks to a python function call. There are two subsections
# which split up the code: the helper function and the target-log-prob
# creation. The helper functions act on data directly and modify objects
# in memory. The target-log-prob function also operates on data (objects)
# but uses closures over it in order to return functions that depend on
# model parameters. This allows them to be used as targets of inference
# algorithms in section 2.

# Helper functions


def pairwise_distance(location_data):
    """Compute pairwise distance matrix between farms

    Args:
        farm_locations_data (float64): Lat-Long coordinates of farms

    Returns:
        float64: tensor of Euclidean distances between entities
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

# Likelihood components

# Regression Component


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

# Infectious pressure


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

# Removal fn


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

# Priors specification


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

# Target log prob


def generate_target_log_prob_fn(log_likelihood_fn, prior_dist_fn):
    """Combine log-likelihood and prior distributions to create the 
    target log prob fn used in inference

    Args:
        log_likelihood_fn (fn): log-likelihood fn of the model
        prior_dist_fn (fn): prior distribution fn of the model parameters

    Returns:
        float64: evaluation of the target log prob

    Notes: 
    The inner fn operates on an unpacked version of the ParameterTuple
    in order to be able to take advantage of partial function 
    evaluations in the functools package. 
    """
    def target_log_prob_eval(regression_param, spatial_param, removal_param):
        # pack up the parameters (lazy way so we don't need to
        # change the structure of the other functions above)
        parameters_tuple = ParameterTuple(regression_param,
                                          spatial_param,
                                          removal_param)
        return log_likelihood_fn(parameters_tuple) + prior_dist_fn(parameters_tuple)
    return target_log_prob_eval

# Section 2 - Probabilistic inference
# This section is dedicated to performing Bayesian infernce given
# a target log prob function. Several MCMC kernels are specified
# with the goal that they work on the same target log prob function.
# Proposal distributions for each are created kernel is created
# based on the shapes initial values of the parameters.

# Proposal distributions


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


@tf.function(jit_compile=True)
def random_walk_metropolis_hastings_baseline(
        target_log_prob_fn, initial_state_tuple, proposal_sigma_tuple,
        num_iter=15, seed=None):
    """Baseline inference scheme for an SIR model. Three kernels are 
    combined back to back to back by iterating over partial states of
    the target log prob function. 

    Args:
        target_log_prob_fn (fn): target function of the inference algorithm
        initial_state_tuple (ParameterTuple): blocks of inference as 
        specified by the parameters to be targeted by each kernel 
        proposal_sigma_tuple (ParameterTuple): parameters of the proposal 
        distributions of each of the blocks of inference 
        num_iter (int, optional): Number of iterations to run the MCMC 
        algo. Defaults to 15.
        seed (int, optional): PRNG seed. Defaults to None.

    Returns:
        parameter_samples (ParameterTuple): MCMC chain for each model parameter
        mcmc_results (ParameterTuple): kernel result for each of model parameter
    """
    # seed handling
    seed = tfp.random.sanitize_seed(
        seed, salt="RandomWalkMetropolisHastings_baseline")

    # Instantiate proposal distributions - a ParameterTuple containing
    # fn's corresponding to the proposal of each block of inference
    proposal_distributions = tf_map(
        lambda x: generate_gaussian_proposal(x), proposal_sigma_tuple)

    # Instantiate the TLP - place holder in case we want to make this
    # depend on other data and want to combine it within the function
    target_log_prob = target_log_prob_fn

    # starting point of the MCMC scheme
    def bootstrap_result(initial_state_tuple):
        """Initialize the kernel trackers

        Args:
            initial_state_tuple (ParameterTuple): inital values for 
            the chains

        Returns:
            ParameterTuple: an instance of the kernel for each of the 
            blocks of inference
        """
        kernel_results = RWMHResult(
            is_accepted=tf.ones_like(target_log_prob(
                *initial_state_tuple), dtype=tf.bool),
            current_state=initial_state_tuple,
            current_state_log_prob=target_log_prob(*initial_state_tuple)
        )
        return ParameterTuple(kernel_results,
                              kernel_results,
                              kernel_results)

    def one_step(current_state, previous_kernel_result, seed):
        """One step of the MCMC algorithm 

        Args:
            current_state (ParameterTuple): current value for each 
            block of inference
            previous_kernel_result (ParameterTuple): outcome of 
            previous kernel containing cached information from 
            the previous iteration
            seed (int): PRNG seed

        Returns:
            current_state (ParameterTuple): new values for the 
            (entire) chain
            kernel_results (ParameterTuple): cached values of the
            intermediate kernel values
        """
        # pick up most recent update log prob
        seeds = tfp.random.split_seed(seed, n=len(
            current_state._fields), salt="one_step")

        current_state_log_prob = previous_kernel_result[-1].current_state_log_prob

        # iterate over each of the blocks we want to update
        intermediate_kernel_results = []

        # TO DO -- Eliminate the for loop over the current state
        # by creating a namedtuple over which maps each field to
        # it's own one_step fn. Each one_step fn instantiates its
        # own proposal fn. Parameters come from a namedtuple over
        # the same fields - i.e. create a fn to *make* a proposal
        # (closure over the variance with the current state
        # providing the mean). End goal is to iterate over a
        # namedtuple fields with signatures
        # ([current_partial_state], [partial_target_log_prob],
        # [proposal_fn]).

        # NEXT TO DO -- unpack the namedtuple of parameters to be
        # arguments for the target_log_prob and take advantage of
        # the partial evaluation fn. We want to make a wrapper
        # one_step fn that acts as a pipe for all the kernels
        # that need to be evaluated in one step of the MCMC

        # IDEA -- kernels can be stacked using tf.scan?

        for field, seed in zip(current_state._fields, seeds):

            # seed handling
            proposal_seed, accept_seed = tfp.random.split_seed(
                seed, n=2, salt="for_loop")

            # pick up partial state (maps to block to perform inference on)
            partial_current_state = getattr(current_state, field)

            #########
            # Propose next step
            proposal_fn = getattr(proposal_distributions, field)
            next_partial_state = proposal_fn(
                partial_current_state, proposal_seed)

            # overright type - Bad for memory I know
            next_complete_state = current_state._replace(
                **{field: next_partial_state})

            #########
            # Compute log accept ratio
            next_target_log_prob = target_log_prob(
                *next_complete_state)

            log_accept_ratio = next_target_log_prob - \
                current_state_log_prob

            # Accept reject step
            log_uniform = tf.math.log(tfp.distributions.Uniform(
                high=tf.constant(1., dtype=DTYPE)).sample(seed=accept_seed))

            is_accepted = log_uniform < log_accept_ratio

            current_state = tf.cond(
                is_accepted, lambda: next_complete_state, lambda: current_state)
            current_state_log_prob = tf.cond(
                is_accepted, lambda: next_target_log_prob,
                lambda: current_state_log_prob)

            new_kernel_results = RWMHResult(
                is_accepted=is_accepted,
                current_state=current_state,
                current_state_log_prob=current_state_log_prob
            )
            intermediate_kernel_results.append(new_kernel_results)
        # this only keeps track of accept/reject of last block
        # new_kernel_results

        return current_state, ParameterTuple(*intermediate_kernel_results)

    # Perform sampling - use the tf.while_loop fn and style
    # Require: body fn and cond fn (i.e. perform body while cond is true)
    # Write the results to the following accumulators

    # Chain values - a ParameterTuple of TensorArrays which expand at each
    # iteration with the next value from the one_step fn
    parameter_samples = tf_map(
        lambda x: tf.TensorArray(dtype=x.dtype, size=num_iter),
        initial_state_tuple)
    # One step result tracker - a ParameterTuple of RMWHResult tuples tracking each
    # sub-kernel outcome (instances of the intermediate_kernel_results)
    mcmc_results = tf_map(
        lambda x: tf.TensorArray(dtype=x.dtype, size=num_iter),
        bootstrap_result(initial_state_tuple))

    def cond(iterator,
             _2,
             _3,
             _4,
             _5,
             _6):
        return iterator < num_iter

    def body(iterator,
             current_state,
             previous_kernel_result,
             parameter_samples,
             mcmc_results,
             seed):
        # Perform one step of in the chain
        this_seed, next_seed = tfp.random.split_seed(seed, n=2, salt="body")
        next_state, next_kernel_result = one_step(
            current_state=current_state,
            previous_kernel_result=previous_kernel_result,
            seed=this_seed
        )

        # Track the outcome - CHAIN STATE
        parameter_samples = tf_map(
            lambda x, a: a.write(iterator, x),
            next_state, parameter_samples)
        # Track the outcome - KERNEL STATE(S)
        mcmc_results = tf_map(lambda x, a: a.write(iterator, x),
                              next_kernel_result, mcmc_results)

        return (iterator + 1,
                next_state,
                next_kernel_result,
                parameter_samples,
                mcmc_results,
                next_seed)

    (_1,
     _2,
     _3,
     parameter_samples,
     mcmc_results,
     _4) = tf.while_loop(cond=cond,
                         body=body,
                         loop_vars=(0,
                                    initial_state_tuple,
                                    bootstrap_result(
                                        initial_state_tuple),
                                    parameter_samples,
                                    mcmc_results,
                                    seed))

    # Formatting
    parameter_samples = tf_map(
        lambda x: x.stack(), parameter_samples)

    mcmc_results = tf_map(lambda x: x.stack(),  mcmc_results)

    return parameter_samples, mcmc_results


def random_walk_metropolis_hastings_kernel(target_log_prob_fn, proposal_scale):
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
            previous_kernel_result.current_state_log_prob

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
        return current_state, kernel_results

    return random_walk_metropolis_hastings_one_step
