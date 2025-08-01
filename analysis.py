import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

a = 1.685 #a^{-1}


#读取数据
def data_load(mu):
    """
    Function load data from two point correlation .dat file
    Args:
        mu(int): mumu ,0,1,2
    Returns:
        C_array(np.array): 2d array,C data each line
        t(np.array): 1d time series 
    """ 
    
    file_list = sorted(glob.glob("./s3/*.dat"))  # 根据你的文件名修改通配符

    real_data_all = []  # 用于存储每个文件的一维 real 数组

    for fname in file_list:
        data = np.loadtxt(fname, comments='#')
        
        filtered = data[data[:, 3] == mu]  # 第4列是 mumu
        
        real_values = filtered[:, 5]
        
        real_data_all.append(real_values)

    # 合并成二维数组，每一行是一个文件的数据
    C_array = np.array(real_data_all)
    t = np.array(range(C_array.shape[1]))

    return C_array,t


def jackknife_method(data):
    N = len(data)
    jk_samples = np.zeros(N)

    for i in range(N):
        jk_samples[i] = np.mean(np.delete(data,i))

    jk_mean = np.mean(jk_samples)
    jk_var = ((N-1)/N) * np.sum((jk_samples - jk_mean)**2)
    jk_error = np.sqrt(jk_var)

    return jk_mean, jk_error


def blocking(data, block_size):
    """
    Block data into blocks of block_size
    Args:
        data: 1d array data to be bined
        block_size: size of the data block
    Returns:
        blocked: 1d array blocked data
    """
    N = np.shape(data)[0]
    n_block = N//block_size

    if N % block_size != 0:
        raise ValueError("Can't be divisible by blocksize")
    
    blocked = np.zeros(n_block)
    for i in range(n_block):
        blocked[i] = np.mean(data[i*block_size:(i+1)*block_size])
    
    return blocked


#effetive mass
def eff_mass(data):
    ratio = data[:,:-1] / data[:,1:]
    m_eff = np.where(ratio > 1,np.log(ratio), np.nan) #choose positive mass value
    return m_eff


#fit function
def fit_func(t,A,m):
    return A*np.exp(-m*t)


def fitting(C_data,t,fit_func,):
    """
    Function to fit the mass parms
    Args:
        C_data: combined C data
        t: time data
        fit_func: function to fit for params
    """
    fit_range=(10,40)
    fit_param = []
    fit_cov = []

    mask = (t >= fit_range[0]) & (t <= fit_range[1])
    t_fit = t[mask]
    C_fit = C_data[:,mask]
    
    for i in range(C_data.shape[0]):
        popt, pcov = curve_fit(fit_func, t_fit, C_fit[i,:])
        fit_param.append(popt)
        fit_cov.append(pcov)

    return np.array(fit_param)


def calc(mu):
    #compute the mass & save them
    C_data,t  = data_load(mu)           #extract data with mumu=0
    C = np.delete(C_data,14,axis=0) #delete the last line for blocking
    t = np.array(range(96))
    
    fit_param = fitting(C,t,fit_func) #calc the fit params
    fit_param[:,1] = fit_param[:,1]*a
    print("successfully compute")
    np.save(f"fit_param_{mu}.npy",fit_param)    #save params to file
    print("successfully saved")
    # print(fit_param.shape)
    # print(f"A={fit_param[:,0]}")
    # print(f"m={fit_param[:,1]}")


def eff_mass_calc(mu):
    #compute effective mass 
    C,t  = data_load(mu)           #extract data with mumu=0
    # C = np.delete(C,14,axis=0) #delete the last line for blocking
    t = np.array(range(95))

    m_eff = eff_mass(C)*a

    return m_eff,t


def plot_eff_mass():
    m_eff_0,t = eff_mass_calc(0)
    m_eff_1,_ = eff_mass_calc(1)
    m_eff_2,_ = eff_mass_calc(2)
    plt.figure()
    plt.scatter(t,m_eff_0[0,:],label='mumu=0')
    plt.scatter(t,m_eff_1[0,:],label='mumu=1')
    plt.scatter(t,m_eff_2[0,:],label='mumu=2')
    plt.grid()
    plt.title("effective mass of different mumu")
    plt.xlabel("t")
    plt.ylabel("m_eff(GeV)")
    plt.legend()
    plt.savefig("effective_mass.png",dpi=200)
    plt.show()   


def analysis(mu):
    """
    Function resampling the fit data with jk & bs
    Args:
        mu(int): mumu, 0,1,2
    Returns:
        None
    """
    data = np.load(f"fit_param_{mu}.npy")
    
    m_block_data = blocking(data[:,1],2)
    A_block_data = blocking(data[:,0],2)
    
    m_mean,m_error = jackknife_method(m_block_data)
    A_mean,A_error = jackknife_method(A_block_data)

    print(f"mumu={mu}")
    print(f"M:mean={m_mean},error={m_error}")
    print(f"A:mean={A_mean},error={A_error}")


if __name__ == "__main__":
    calc(0)
    calc(1)
    calc(2)
    analysis(0)
    analysis(1)
    analysis(2)

    

    

    


    




