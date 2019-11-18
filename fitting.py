from AbfAnalysis import *
from helpers import *
from lmfit import Model
import numpy as np
import statistics

def linear(t,m,y0):
    return m*t+y0


def first_oder_sys_response(t,y0,y_ss,tau):
    return (y0-y_ss)*np.exp(-t/tau) + y_ss


def two_first_oder_sys_responses(t,y0_1,y0_2,y_ss_1,y_ss_2,tau_1,tau_2):
    return (y0_1-y_ss_1)*np.exp(-t/tau_1) + (y0_2-y_ss_2)*np.exp(-t/tau_2) + y_ss_1 + y_ss_2


# def exponential(x,A,k,C):
#     return A*np.exp(k*x) + C
#
#
# def biexponential(x,A1,A2,k1,k2,C1,C2):
#     return  A1*np.exp(k1*x) + A2*np.exp(k2*x) + C1 + C2


def guess_init_vals(x,y,function_name):
    """
    guesses initial values for different functions
    :param x: np array of x values
    :param y: np array of x values
    :param function_name: 'linear' / 'exponential' / 'biexponential'
    :return: a dictionary with the initial values
    """
    def linear_guess(x,y):
        x_first_half, x_second_half = np.array_split(x, 2)
        y_first_half, y_second_half = np.array_split(y, 2)
        avg_delta_x = np.average(x_second_half) - np.average(x_first_half)
        avg_delta_y = np.average(y_second_half) - np.average(y_first_half)
        estimated_m = avg_delta_y / avg_delta_x
        first_guess_y0 = y[0] - estimated_m * x[0]
        second_guess_y0 = y_second_half[0] - estimated_m * x_second_half[0]
        third_guess_y0 = y[-1] - estimated_m * x[-1]
        y0_guesses = [first_guess_y0, second_guess_y0, third_guess_y0]
        estimated_y0 = statistics.mean(y0_guesses)
        return estimated_m, estimated_y0

    if function_name == 'linear':
        estimated_m, estimated_y0 = linear_guess(x,y)
        return estimated_m, estimated_y0
    if function_name == 'exponential':
        estimated_linear_slope ,estimated_linear_y0 = linear_guess(x,y)
        estimated_y_ss = estimated_linear_slope*4*x[-1]+estimated_linear_y0
        estimated_tau = (estimated_y_ss-estimated_linear_y0)/estimated_linear_slope
        return estimated_linear_y0, estimated_y_ss, estimated_tau
    else:
        raise NotImplementedError('this function was not implemented for the function_name:', function_name)


def fit_linear(x,y, make_plot=False):
    init_m, init_y0 = guess_init_vals(x, y, 'linear')
    fit_model = Model(linear)
    result = fit_model.fit(y, t=x, m=init_m, y0=init_y0)

    if make_plot:
        print(result.fit_report())
        plt.plot(x, y, 'bo')
        plt.plot(x, result.init_fit, 'k--', label='initial fit')
        plt.plot(x, result.best_fit, 'r-', label='best fit')
        plt.legend(loc='best')
        plt.show()

    return result


def fit_exponential(x,y, make_plot=False):
    y0, y_ss, tau = guess_init_vals(x, y, 'exponential')
    fit_model = Model(first_oder_sys_response)
    fit_model.set_param_hint('tau', value=tau, min=0, max=20)
    params = fit_model.make_params(y0=y0, y_ss=y_ss)
    result = fit_model.fit(y, params, t=x)
    if result.redchi > 5:
        raise Warning('The reduced chi of this fit is bigger than 5. Chi =', result.redchi)
    print(result.best_fit)
    if make_plot:
        print(result.fit_report())
        plt.plot(x, y, 'bo')
        plt.plot(x, result.init_fit, 'k--', label='initial fit')
        plt.plot(x, result.best_fit, 'r-', label='best fit')
        plt.legend(loc='best')
        plt.show()

    return result


def fit_biexponential(x,y,y0_1,y0_2,y_ss_1,y_ss_2,tau_1,tau_2, make_plot=False):
    fit_model = Model(two_first_oder_sys_responses)
    result = fit_model.fit(y, t=x, y0_1=y0_1, y0_2=y0_2, y_ss_1=y_ss_1, y_ss_2=y_ss_2, tau_1=tau_1, tau_2=tau_2)

    if make_plot:
        print(result.fit_report())
        plt.plot(x, y, 'bo')
        plt.plot(x, result.best_fit, 'r-', label='best fit')
        plt.legend(loc='best')
        plt.show()

    return result


def fit_pre_light(sweep, fit_type, t0=None, make_plot=True):
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
        return fit_exponential(fit_time, fit_current,make_plot=make_plot)
    else:
        raise NotImplementedError('this function was not implemented for the fit_type:', fit_type)


def get_r_squared_from_fit_results(fit_results):
    ss_res = np.sum(fit_results.residual ** 2)
    ss_tot = np.sum((fit_results.data - np.mean(fit_results.data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def estimate_data_with_fit(t, function, fit_result):
    fit_result_values = fit_result.best_values
    y = np.zeros(shape=t.shape)
    if function == 'exponential':
        y0 = fit_result_values['y0']
        y_ss = fit_result_values['y_ss']
        tau = fit_result_values['tau']
        for i in range(len(t)):
            y[i] = first_oder_sys_response(t[i], y0, y_ss, tau)
    return y


# abfToAnalyze = '/home/kormanav/Dokumente/TEVC_13_11_2019/2019_11_13_0085.abf'

abfToAnalyze = '/Volumes/Transcend/TEVC_13_11_2019/2019_11_13_0085.abf'

abf = ActiveAbf(abfToAnalyze)
sweepnr = 4
sweep1 = sweep(abfToAnalyze, sweepnr, [400, 16000])
plot_sweep(sweep1)
pre_light_fit_result = fit_pre_light(sweep1, 'exponential')
t = np.linspace(0, 3.0, num=5000)
y = estimate_data_with_fit(t,'exponential',pre_light_fit_result)
print(t, y)
plt.plot(t, y, 'bo')
plt.show()