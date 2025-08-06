#home brewed jackknife & bootstrap
#@ixhong
#since 2025.8
import numpy as np

DEFAULTSEED = 7271987


def my_bootstrap(data,times):
    """
    my_bootstrap(data,times)
    Args:
        data:2d array,one configration per line
        times: times you want to bootstrap resample
    Return:
        mean,error:1d array
    """
    N,T = data.shape
    bs_samples = np.zeros((times,T))

    for i in range(times):
        #change the random method as in Analysistoolbox 
        rng = np.random.default_rng(DEFAULTSEED+i)
        chosen = rng.integers(0,N,N)
        bs_samples[i] = np.mean(data[chosen, :],axis=0)

    mean = np.mean(bs_samples, axis=0)
    variance = np.sum((bs_samples - mean) ** 2, axis=0) / (times)
    # error = np.sqrt(variance)
    error = np.std(bs_samples, axis=0, ddof=1)

    return mean, error


def my_jackknife(data):
    """
    my_jackknife(data)
    Args:
        data:2d array
    Return:
        jk_mean,jk_error
    """
    N,T = data.shape
    jk_samples = np.zeros((N,T))

    for i in range(N):
        jk_samples[i] = np.mean(np.delete(data,i,axis=0),axis=0)
    
    jk_mean = np.mean(jk_samples,axis=0)
    jk_err = np.sqrt((N - 1) * np.mean((jk_samples - jk_mean[np.newaxis, :])**2, axis=0))

    return jk_mean,jk_err


def data_gen(N,T):
    #generate mock data
    """
    Args:
        N: number of configurations
        T: data number per conf
    Return:
        true_data: theoritecal data
        data: theoritecal data with noise added
        t: just self variable
    """
    T = 4
    m = 0.5  
    A = 1.0  
    N_cfg = N  
    noise_level = 0.8 

    t = np.linspace(0,5,T)

    true_data = A * (np.exp(-m * t))

    data = np.zeros((N_cfg, T))
    for i in range(N_cfg):
        noise = np.random.default_rng(i).normal(loc=0.0, scale=noise_level, size=T)
        data[i] = true_data + noise

    return true_data,data,t






