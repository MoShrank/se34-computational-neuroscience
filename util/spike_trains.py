import numpy as np


def get_poisson_spike_train(
    rate: float, no_spikes: int, no_spike_trains: int, dt: float, seed=None
) -> np.ndarray:
    """
    A function that generates a no_spike_trains * no_spikes spike_train matrix

    Args:
        rate (float): rate of the Poisson process
        no_spikes (int): number of spikes per spike_train
        no_spike_trains (int): number of spike_trains
        dt (float): time step
        seed (int): random seed
    Returns:
        spike_train (np.array): spike_train matrix
    """
    if seed:
        np.random.seed(seed=seed)

    # generate uniformly distributed random variables
    u_rand = np.random.rand(no_spikes, no_spike_trains)

    # generate Poisson train
    poisson_train = 1.0 * (u_rand < rate * (dt / 1000.0))

    return poisson_train
