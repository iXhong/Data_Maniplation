#Compare my jackknife & bootstrap results with functions in analysistoolbox
#@ixhong
#since
from latqcdtools.testing import print_results,concludeTest
from latqcdtools.statistics.bootstr import bootstr
from latqcdtools.statistics.jackknife import jackknife as jackk
from latqcdtools.base.initialize import DEFAULTSEED
import latqcdtools.base.logger as logger
import numpy as np
import matplotlib.pyplot as plt
from minefunctions import my_bootstrap,my_jackknife,data_gen


EPSILON = 1e-3


def simple_method(data):
    N = len(data)
    mean = np.mean(data,axis=0)
    std_dev = np.std(data,axis=0)
    error = std_dev/np.sqrt(N-1)
    return mean,error


def func(data):
    return np.mean(data,axis=0)


if __name__ == '__main__':
    lpass = True

    _,data,t = data_gen(1000,4)
    #my functions
    mbs_mean,mbs_error = my_bootstrap(data,1000)
    mjk_mean,mjk_error = my_jackknife(data)
    #standard
    mean,error = simple_method(data)
    #analysistoolbox
    ajk_mean,ajk_error = jackk(func,data,1000,conf_axis=0)
    abs_mean,abs_error = bootstr(func,data,numb_samples=1000,seed=DEFAULTSEED,nproc=1,return_sample=False,conf_axis=0)

    jk_diff_mean = np.abs(mjk_mean - ajk_mean)
    jk_diff_error = np.abs(mjk_error - ajk_error)
    bs_diff_mean = np.abs(mbs_mean-abs_mean)
    bs_diff_error = np.abs(mbs_error - ajk_error)

    lpass *= print_results(mjk_mean,ajk_mean,mjk_error,ajk_error,"jackknife: mine func vs analysistoolbox",EPSILON)
    lpass *= print_results(mbs_mean,abs_mean,mbs_error,abs_error,"bootstrap: minefunc vs analysis",EPSILON)

    concludeTest(lpass)


    print(f"mbs_mean={mbs_mean},mbs_error={mbs_error}")
    print(f"abs_mean={mbs_mean},abs_error={abs_error}")
    print(f"mjk_mean={mjk_mean},mjk_error={mjk_error}")
    print(f"ajk_mean={ajk_mean},ajk_error={ajk_error}")
    # print(f"mean={mean},jk_error={error}")
    print(f"bootstr mean diff={bs_diff_mean}")
    print(f"bootstr error diff={bs_diff_error}")
    print(f"jackk mean diff={jk_diff_mean}")
    print(f"jackk error diff={jk_diff_error}")
    


    