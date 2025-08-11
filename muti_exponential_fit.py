import numpy as np
from scipy.optimize import curve_fit
from myfunctions import data_load,ref_data
import matplotlib.pyplot as plt


### get jackknife samples
def jackknife_samples(data):

    N,T = data.shape
    jk_samples = np.zeros((N,T))

    for i in range(N):
        jk_samples[i] = np.mean(np.delete(data,i,axis=0),axis=0)

    t_vals = np.array(range(T))
    return jk_samples,t_vals


#fit function
# def fit_func_exp(t,A0,m0,A1,m1):
#     return A0*np.exp(-t*m0)+A1*np.exp(-t*m1)
def fit_func_exp(t,A0,m0):
    return A0*np.exp(-t*m0)
    # return A0*np.cosh(m0*(48-t))+A1*np.cosh(m1*(48-t))
def fit_func_cosh(t, A, m0, T):
    return A * np.cosh(m0 * (t-T/2))


#fitting
def fitting(jk_samples,t,t_min,t_max):
    n = jk_samples.shape[0]
    mass = np.zeros(n)

    t_fit = t[(t>=t_min) & (t<=t_max)]

    for i in range(n):
        C_fit = jk_samples[i,t_fit]
        p0 = [C_fit[0],0.5]
        try:
            # popt,_ = curve_fit(lambda t, A0, m0: fit_func_cosh(t, A0, m0,T=48),t_fit,C_fit,p0)
            popt,_ = curve_fit(fit_func_exp,t_fit,C_fit,p0)
            mass[i] = popt[1]
        except RuntimeError:
            mass[i] = np.nan

    return mass


#jackknife method
def jackknife(data):
    N = len(data)
    jk_samples = np.zeros(N)
    for i in range(N):
        jk_samples[i] = np.mean(np.delete(data,i))
    jk_mean = np.mean(jk_samples)
    jk_error = np.sqrt(N-1)*np.std(jk_samples,ddof=0)

    return jk_mean,jk_error


#try different t_min
def scan_tmin(tmin_l,tmin_r,t_max,jk_samples,t_vals):
    t_min_list = range(tmin_l,tmin_r)
    mass_values = []
    mass_errors = []

    for t_min in t_min_list:
        mass = fitting(jk_samples,t_vals,t_min,t_max)
        mean,error = jackknife(mass)
        mass_values.append(mean)
        mass_errors.append(error)

    return np.array(mass_values),np.array(mass_errors),t_min_list



if __name__ == "__main__":

    # t_min_list = range(5,15)
    t_max = 35
    t_min_l = 1
    t_min_r = 15

    # data,t = data_load(0,"./s3/mass/")
    data,t = ref_data("s3/mass-0.5706655/corr.dat")

    aver_data = (data[:,:48] + np.flip(data[:,-48:]))/2

    jk_samples,t_vals = jackknife_samples(aver_data)

    # mass = fitting(jk_samples,t_vals,t_min,t_max)
    mass_values,mass_errors,t_min_list = scan_tmin(t_min_l,t_min_r,t_max,jk_samples,t_vals)
    print(mass_values)

    # plt.figure(figsize=(10,7))
    # plt.title(f"fit mass under different t_min(t_max={t_max})")
    # plt.errorbar(t_min_list,mass_values,mass_errors,fmt='o',label="mass errorbar")
    # plt.xlabel("t_min")
    # plt.ylabel("mass_value")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # plt.savefig(f"mass_tmins_{t_max}.png",dpi=300)
    plt.figure(figsize=(8,7))
    plt.title(f"fit mass under different t_min(t_max={t_max})")

    # 创建第一个y轴
    ax1 = plt.gca()  # 获取当前轴

    # 绘制主数据和误差条
    ax1.errorbar(t_min_list, mass_values, yerr=mass_errors, fmt='o', 
                label="mass value ± error", color='tab:blue')
    ax1.set_xlabel("t_min")
    ax1.set_ylabel("mass_value", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid()

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制误差值曲线
    ax2.plot(t_min_list, mass_errors, 's--', color='tab:red', 
            label="error value")
    ax2.set_ylabel("error value", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()  # 调整布局防止标签重叠
    plt.savefig(f"exp_mass_tmins_{t_max}.png", dpi=300)
    plt.show()

    # mass_values,mass_errors,t_min_list=scan_tmin(5,15,30,jk_samples,t_vals,[0.1,0.1,0.1,0.1])
    # np.savez("mass.npz",mass_values=mass_values,mass_errors=mass_errors,t_min_list=t_min_list)



    
