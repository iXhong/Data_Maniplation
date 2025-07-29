import numpy as np
import matplotlib.pyplot as plt


#blocking
def blocking(data,block_size):

    N = len(data)
    M = N//block_size
    blocked_data = np.zeros((M,block_size))
    for i in range(M):
        blocked_data[i,:] = np.array(data[i*block_size:(i+1)*block_size])

    return blocked_data


#Jackknife blocking 
def jackknife_block(data,func):
    '''
    jackknife method
    Args:data,func
    return:
         origin,jack_variance,jack_error
    '''
    N, M = np.shape(data)

    origin_mean = np.mean(data)
    jackknife_estimate = np.zeros(N)

    for i in range(N):
        jackknife_sample = np.delete(data,i,axis=0).reshape(-1)
        jackknife_estimate[i] = func(jackknife_sample)

    jack_mean = np.mean(jackknife_estimate)
    jack_variance = ((N-1)/N) * np.sum((jackknife_estimate - jack_mean)**2)
    jack_error = np.sqrt(jack_variance)

    return origin_mean,jack_variance,jack_error


#jackknife 1d
def jackknife_1d(data,func):
    N = len(data)
    origin_mean = np.mean(data)
    jackknife_estimate = np.zeros(N)

    for i in range(N):
        jackknife_sample = np.delete(data,i)
        jackknife_estimate[i] = func(jackknife_sample)

    jack_mean = np.mean(jackknife_estimate)
    jack_variance = ((N-1)/N) * np.sum((jackknife_estimate - jack_mean)**2)
    jack_error = np.sqrt(jack_variance)

    return origin_mean,jack_variance,jack_error


#statistic function
def func(data):
    return np.mean(data,axis=0)


#test
N = 100
mu = 5.0
sigma = 1.0
np.random.seed(42) 
mock_data = np.random.normal(loc=mu, scale=sigma, size=N)


theoretical_error = sigma / np.sqrt(N)


r = blocking(mock_data,5)


# mean, jk_vals, jk_error = jackknife_block(r,func)
mean, jk_vals, jk_error = jackknife_1d(r,func)


print(f"样本平均值: {mean:.4f}")
print(f"理论标准误差: {theoretical_error:.4f}")
print(f"Jackknife 误差估计: {jk_error:.4f}")