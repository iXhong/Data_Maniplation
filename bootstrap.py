import numpy as np


#Blocking data
def blocking(data,block_size):

    N = len(data)
    M = N//block_size
    blocked_data = np.zeros((M,block_size))
    for i in range(M):
        blocked_data[i,:] = np.array(data[i*block_size:(i+1)*block_size])

    return blocked_data


#Bootstrap method
def bootstrap_method(data,func,times):

    N = len(data)
    bootstrp_stats = np.zeros(times)

    for i in range(times):
        sample = np.random.choice(data,N,replace=True)
        stat = func(sample)
        bootstrp_stats[i] = stat

    stats_mean = np.mean(bootstrp_stats)
    stats_variance = 1/times * np.sum((bootstrp_stats - stats_mean)**2)
    bias = np.sqrt(stats_variance)

    return stats_mean,bias


#Bootstrap block
def bootstrap_block(data,func,times):
    """
    bootstrap on 2D data
    
    """

    N = np.shape(data)[0]
    bootstrap_estimate = np.zeros(times)
    raws = np.arange(N)

    for i in range(times):
        raws_choosed = np.random.choice(raws,N,replace=True)
        sample = data[raws_choosed,:].reshape(-1)
        bootstrap_estimate[i] = func(sample)

    bootstrap_mean = np.mean(bootstrap_estimate)
    bootstrap_variance = 1/times * np.sum((bootstrap_estimate - bootstrap_mean)**2)
    bias = np.sqrt(bootstrap_variance)

    return bootstrap_mean,bias


def func(data):
    return np.mean(data)


#test part
N = 100
mu = 5.0
sigma = 1.0
np.random.seed(42) 
data = np.random.normal(loc=mu, scale=sigma, size=N)
# mock_data = blocking(data,5)


theoretical_error = sigma / np.sqrt(N)

mean,bias = bootstrap_block(data,func,1000)

print(f"样本平均值: {mean:.4f}")
print(f"理论标准误差: {theoretical_error:.4f}")
print(f"bootstrap 误差估计: {bias:.4f}")
     