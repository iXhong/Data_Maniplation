import numpy as np
from scipy.optimize import curve_fit
from myfunctions import data_load
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
def fit_func(t,A0,m0,A1,m1):
    return A0*np.exp(-t*m0)+A1*np.exp(-t*m1)
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
        p0 = [C_fit[len(C_fit)//2],0.5]
        try:
            popt,_ = curve_fit(lambda t, A0, m0: fit_func_cosh(t, A0, m0,T=48),t_fit,C_fit,p0)
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

    return mass_values,mass_errors,t_min_list



if __name__ == "__main__":

    # t_min_list = range(5,15)
    t_max = 35
    t_min = 15

    data,t = data_load(0,"./s3/mass/")

    aver_data = (data[:,:48] + np.flip(data[:,-48:]))/2

    jk_samples,t_vals = jackknife_samples(aver_data)

    mass = fitting(jk_samples,t_vals,t_min,t_max)

    print(mass)
    

    # mass_values,mass_errors,t_min_list=scan_tmin(5,15,30,jk_samples,t_vals,[0.1,0.1,0.1,0.1])
    # np.savez("mass.npz",mass_values=mass_values,mass_errors=mass_errors,t_min_list=t_min_list)



    
