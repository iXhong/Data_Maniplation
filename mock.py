import numpy as np
import matplotlib.pyplot as plt


def data_gen(N):
    # 参数
    T = 40
    m = 0.5  # 模拟质量
    A = 1.0  # 振幅
    N_cfg = N  # 配置数
    noise_level = 0.8  # 噪声比例

    t_vals = np.linspace(0,5,T)

    true_C = A * (np.exp(-m * t_vals))

    # 生成 N_cfg 个配置，每个加上随机噪声
    C_data = np.zeros((N_cfg, T))
    for i in range(N_cfg):
        noise = np.random.default_rng(i).normal(loc=0.0, scale=noise_level, size=T)
        C_data[i] = true_C + noise

    return true_C,C_data,t_vals


def blocking(data,block_size):
    N, T = data.shape

    n_block = N//block_size
    if N % block_size != 0:
        raise ValueError("配置数量不能整除 block_size")
    
    blocked = np.zeros((n_block, T))
    for i in range(n_block):
        blocked[i] = np.mean(data[i*block_size:(i+1)*block_size], axis=0)
    return blocked


def jackknife(data):

    N,T = data.shape
    jk_samples = np.zeros((N,T))

    for i in range(N):
        jk_samples[i] = np.mean(np.delete(data,i,axis=0),axis=0)
    
    jk_mean = np.mean(jk_samples,axis=0)
    jk_err = np.sqrt((N - 1) * np.mean((jk_samples - jk_mean[np.newaxis, :])**2, axis=0))

    return jk_mean,jk_err


def bootstrap(data,times):
    
    N,T = data.shape
    bs_samples = np.zeros((times,T))
    indices = np.arange(N)

    for i in range(times):
        chosen = np.random.choice(indices, N, replace=True)
        bs_samples[i] = np.mean(data[chosen, :],axis=0)

    mean = np.mean(bs_samples, axis=0)
    variance = np.sum((bs_samples - mean) ** 2, axis=0) / (times - 1)
    error = np.sqrt(variance)

    return mean, error


def simple_method(data):
    N = len(data)
    mean = np.mean(data,axis=0)
    std_dev = np.std(data,axis=0)
    error = std_dev/np.sqrt(N)
    return mean,error


"""
Test part
"""

if __name__ == "__main__":
    N=1000
    times = 2000
    true_data,noise_data,t = data_gen(N)


    mean = np.mean(noise_data,axis=0)
    error = np.std(noise_data,axis=0)

    jk_mean,jk_error = jackknife(noise_data)
    bs_mean,bs_error = bootstrap(noise_data,times)


    plt.figure(figsize=(10,7))
    # plt.scatter(t_vals,C_data[0,:],label="C with noise")
    plt.errorbar(t,mean,error,fmt='o',label="simple mean & error")
    # plt.errorbar(t,jk_mean,jk_error,fmt='o',label="jackknife measurment")
    # plt.errorbar(t,bs_mean,bs_error,fmt='.',label="bootstrap measurement")
    plt.title(f"simple method,N_cfg={N}")
    plt.xlabel('t')
    plt.ylabel("y")
    plt.plot(t,true_data,label="theory: y=e^(-0.5*t)")
    # plt.savefig(f"bootstrap_N={N}_times={times}.png",dpi=300)
    plt.legend()
    plt.grid()
    plt.show()










