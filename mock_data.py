import numpy as np
import matplotlib.pyplot as plt

def exp_decay(t):
    #data function
    A = 10.0
    m = 1.0

    return A*np.exp(-m*t)


def data_generate(func,x,seed):
    #add normal distribution noise on func data
    y = func(x)
    np.random.seed(seed)
    mu, sigma = 0, 5
    noise = np.random.normal(mu,sigma,y.shape)
    data = noise + y

    return data


def data_list(func,x,nums):
    #generate list of data with noise
    seeds = np.arange(nums)
    cols = len(x)
    rows = nums
    data = np.zeros((nums,cols))
    for i in np.arange(nums):
        data[i,:] =  data_generate(func,x,seeds[i])

    return data


#jackknife
def jackknife(data):
    N, M = np.shape(data)
    origin_mean = np.mean(data, axis=0)
    jack_esti = np.zeros((N, M))

    for i in range(N):
        jack_sample = np.concatenate([data[:i], data[i+1:]], axis=0)
        jack_esti[i, :] = np.mean(jack_sample, axis=0)

    jack_mean = np.mean(jack_esti, axis=0)
    jack_variance = ((N - 1) / N) * np.sum((jack_esti - jack_mean)**2, axis=0)
    jack_error = np.sqrt(jack_variance)

    return jack_mean, jack_error


#bootstrap
def bootstrap(data, times):
    N, M = np.shape(data)
    bootstrap_esti = np.zeros((times, M))
    indices = np.arange(N)

    for i in range(times):
        chosen = np.random.choice(indices, N, replace=True)
        sample = data[chosen, :]
        bootstrap_esti[i, :] = np.mean(sample, axis=0)

    mean = np.mean(bootstrap_esti, axis=0)
    variance = np.sum((bootstrap_esti - mean) ** 2, axis=0) / (times - 1)
    error = np.sqrt(variance)

    return mean, error





#test part
x = np.linspace(0,10,100)

mock_data = data_list(exp_decay,x,100)


# mean, error = jackknife(mock_data)
mean, error = bootstrap(mock_data,10)



# for i in range(mock_data.shape[1]):
#     print(f"列 {i}: mean = {mean[i]:.4f} ± {error[i]:.4f}")

plt.figure()
plt.errorbar(x,mean,error,fmt='o', capsize=3, label='mean ± error')
# for i in range(np.shape(mock_data)[0]):
#     plt.scatter(x,mock_data[i,:])

plt.show()

