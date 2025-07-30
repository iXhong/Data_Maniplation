import numpy as np
import matplotlib.pyplot as plt


def data_gen():
    # 参数
    T = 32  # 时间长度
    m = 0.5  # 模拟质量
    A = 1.0  # 振幅
    N_cfg = 10  # 配置数
    noise_level = 0.02  # 噪声比例

    # 时间点
    t_vals = np.arange(T)

    # 真实函数形状（无噪声）
    true_C = A * (np.exp(-m * t_vals) - np.exp(-m * (T - t_vals)))

    # 生成 N_cfg 个配置，每个加上随机噪声
    C_data = np.zeros((N_cfg, T))
    for i in range(N_cfg):
        noise = np.random.normal(loc=0.0, scale=noise_level, size=T)
        C_data[i] = true_C * (1 + noise)

    return C_data,t_vals


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


def eff_mass(data):
    m_eff = np.log(data[:,:-1] / data[:,1:])
    return m_eff


# data = np.loadtxt("./s3/TwoPt_ss_z2_conf210_mom000.dat",comments='#')
# t = data[0:95, 4].astype(int)
# C_real = data[0:95, 5]
C_data,t = data_gen()
m_data = eff_mass(C_data)

# data = blocking(m_data,2)
# mean,error = jackknife(data)

# print(f"mean={mean},error={error}")

# m_eff = eff_mass(C)






