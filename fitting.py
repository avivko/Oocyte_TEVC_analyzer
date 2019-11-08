from AbfAnalysis import *
from helpers import *
import numpy as np


def linear(x,m,y0):
    return m*x+y0


def exponential(x,A,k,C):
    return A*np.exp(k*x) + C


def biexponential(x,A1,A2,k1,k2,C1,C2):
    return  A1*np.exp(k1*x) + A2*np.exp(k2*x) + C1 + C2


def fit_linear(x,y,init_m,init_y0, make_plot=False):
    fit_model = Model(linear)
    result = fit_model.fit(y, x=x, m=init_m, y0=init_y0)

    if make_plot:
        print(result.fit_report())
        plt.plot(x, y, 'bo')
        plt.plot(x, result.best_fit, 'r-', label='best fit')
        plt.legend(loc='best')
        plt.show()

    return result


def fit_exponential(x,y,init_A,init_k,init_C, make_plot=False):
    fit_model = Model(exponential)
    result = fit_model.fit(y, x=x, A=init_A, k=init_k, C=init_C)

    if make_plot:
        print(result.fit_report())
        plt.plot(x, y, 'bo')
        plt.plot(x, result.best_fit, 'r-', label='best fit')
        plt.legend(loc='best')
        plt.show()

    return result


def fit_biexponential(x,y,init_A1,init_A2,init_k1,init_k2,init_C1,init_C2, make_plot=False):
    fit_model = Model(exponential)
    result = fit_model.fit(y, x=x, A1=init_A1, A2=init_A2, k1=init_k1, k2=init_k2, C1=init_C1, C2=init_C2)

    if make_plot:
        print(result.fit_report())
        plt.plot(x, y, 'bo')
        plt.plot(x, result.best_fit, 'r-', label='best fit')
        plt.legend(loc='best')
        plt.show()

    return result



def fit_pre_light(sweep, fit_type, init_param_dict, t0=None):
    sweep_data = sweep.get_sweep_data()
    sweep_times = sweep_data['times']
    sweep_currents = sweep_data['currents']
    if t0 is None: # if starting time for fit is not specified
        t0 = sweep_times[0]
        assert t0 != 0, 'the first fit should not start at t0 due to the capacitance peak'
    recorded_t0 = get_closest_value_from_ordered_array(t0,sweep_times)
    t0_index = get_index_of_unique_value(recorded_t0,sweep_times)
    t_light_on = sweep_data['shutter on']
    recorded_t_light_on = get_closest_value_from_ordered_array(t_light_on,sweep_times)
    t_light_on_index = get_index_of_unique_value(recorded_t_light_on, sweep_times)
    fit_time = sweep_times[t0_index:t_light_on_index]
    fit_current = sweep_currents[t0_index:t_light_on_index]
    if fit_type == 'exponential':
        init_A = init_param_dict['init_A']
        init_k = init_param_dict['init_k']
        init_C = init_param_dict['init_C']
        fit_exponential(fit_time,fit_current,init_A,init_k,init_C,make_plot=True)

abfToAnalyze = '/home/kormanav/Dokumente/TEVC_29_10_2019/2019_10_29_0053.abf'

abf = ActiveAbf(abfToAnalyze)
sweep1=sweep(abfToAnalyze,1,[400,4500])
plot_sweep(sweep1)
initial_fit_parameters = {
    'init_A': 1000,
    'init_k': -6,
    'init_C': 10
}
fit_pre_light(sweep1,'exponential',initial_fit_parameters)
#sweep1slice = sweep(abfToAnalyze,2,[500,2000])
# sweep1data = sweep1slice.get_sweep_data()
# time = sweep1data['times']
# current = sweep1data['currents']
# fit_linear(time, current, 0.01, 300)
# fit_exponential(time, current, 1000, -6, 500)
# plot_sweep(sweep1)