import numpy as np
import numpy.random as rnd


def latin_hypercube_uniform(dim,n_samples,low=0,high=1):
    '''
    generates samples with latin hypercube sampling. 
    
    dim: int, the dimension
    n_samples: int, the number of samples
    low: int or tuple/array, the left interval end
    high: int or tuple/array, the right interval end

    returns the samples
    '''
    if type(low) == int or type(low) == float:
        low = np.ones(dim) * low 
        high = np.ones(dim) * high
    sample_space = np.zeros((dim, n_samples))
    rng = np.random.default_rng()
    for d in range(dim):
        sample_cats = np.linspace(low[d], high[d], n_samples, endpoint=False)
        sample = (sample_cats + abs((high[d]-low[d])/n_samples) * rnd.random_sample())
        rng.shuffle(sample)
        sample_space[d] = sample
    return sample_space.T