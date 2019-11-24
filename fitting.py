import matplotlib.patches as mpatches
import matplotlib as plt
from AbfAnalysis import *
from helpers import *
from lmfit import Model
import numpy as np
import statistics
import warnings


def linear(t, m, y0):
    return m * t + y0


def first_oder_sys_response(t, y0, y_ss, tau):
    return (y0 - y_ss) * np.exp(-t / tau) + y_ss


def two_first_oder_sys_responses(t, y0_1, y0_2, y_ss_1, y_ss_2, tau_1, tau_2):
    return (y0_1 - y_ss_1) * np.exp(-t / tau_1) + (y0_2 - y_ss_2) * np.exp(-t / tau_2) + y_ss_1 + y_ss_2


def get_r_squared_from_fit_results(fit_results):
    ss_res = np.sum(fit_results.residual ** 2)
    ss_tot = np.sum((fit_results.data - np.mean(fit_results.data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def guess_init_vals(x, y, function_name):
    """
    guesses initial values for different functions
    :param x: np array of x values
    :param y: np array of x values
    :param function_name: 'linear' / 'exponential'
    :return: a dictionary with the initial values
    """

    def linear_guess(x, y):
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

    def linear_guess_from_zero(x, y):
        avg_x = np.average(x)
        avg_y = np.average(y)
        estimated_m = avg_y / avg_x
        return estimated_m

    if function_name == 'linear':
        estimated_m, estimated_y0 = linear_guess(x, y)
        return estimated_m, estimated_y0
    if function_name == 'linear from zero':
        estimated_m = linear_guess_from_zero(x, y)
        return estimated_m
    if function_name == 'exponential':
        estimated_linear_slope, estimated_linear_y0 = linear_guess(x, y)
        estimated_y_ss = estimated_linear_slope * 4 * x[-1] + estimated_linear_y0
        estimated_tau = (estimated_y_ss - estimated_linear_y0) / estimated_linear_slope
        return estimated_linear_y0, estimated_y_ss, estimated_tau
    if function_name == 'exponential from zero':
        estimated_linear_slope = linear_guess_from_zero(x, y)
        estimated_y_ss = estimated_linear_slope * 4 * x[-1]
        estimated_tau = estimated_y_ss / estimated_linear_slope
        return estimated_y_ss, estimated_tau
    else:
        raise NotImplementedError('this function was not implemented for the function_name:', function_name)


def plot_fit(x, y, fit_result):
    print(fit_result.fit_report())
    fig, ax = plt.subplots()
    ax.plot(x, y, 'bo')
    ax.plot(x, fit_result.init_fit, 'k--', label='initial fit')
    ax.plot(x, fit_result.best_fit, 'r-', label='best fit')
    ax.legend(loc='best')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label='red_chi^2 = ' + str(truncate(fit_result.redchi, 2))))
    handles.append(
        mpatches.Patch(color='none', label='R^2 = ' + str(truncate(get_r_squared_from_fit_results(fit_result), 2))))
    ax.legend(handles=handles)
    plt.show()


def fit_linear(x, y, make_plot=False):
    init_m, init_y0 = guess_init_vals(x, y, 'linear')
    fit_model = Model(linear)
    params = fit_model.make_params(y0=init_y0, m=init_m)
    result = fit_model.fit(y, params, t=x)

    if make_plot:
        plot_fit(x, y, result)

    return 'linear', result


def fit_exponential(x, y, fixed_y0=None, t_shift=0, make_plot=False):
    fit_model = Model(first_oder_sys_response)
    if fixed_y0 is None:
        y0, y_ss, tau = guess_init_vals(x, y, 'exponential')
        fit_model.set_param_hint('tau', value=tau, min=0, max=60)
        params = fit_model.make_params(y0=y0, y_ss=y_ss)
    elif fixed_y0 is not None:
        if t_shift == 0:
            y_ss, tau = guess_init_vals(x, y, 'exponential from zero')
            fit_model.set_param_hint('tau', value=tau, min=0, max=60)
            fit_model.set_param_hint('y0', value=fixed_y0, vary=False)
            params = fit_model.make_params(y_ss=y_ss)
        elif t_shift > 0:
            x = x - t_shift
            y_ss, tau = guess_init_vals(x, y, 'exponential from zero')
            fit_model.set_param_hint('tau', value=tau, min=0, max=60)
            fit_model.set_param_hint('y0', value=fixed_y0, vary=False)
            params = fit_model.make_params(y_ss=y_ss)
        else:
            raise ValueError('t_shift should be the time when the shutter is turned on (>= 0) but is:', str(t_shift))
    else:
        raise ValueError('starting_from_0 should be None/ the value of the fixed y0. is: ' + str(fixed_y0))
    result = fit_model.fit(y, params, t=x)
    if make_plot:
        plot_fit(x, y, result)
    if result.redchi > 15:
        warnings.warn('The reduced chi of the exponential fit is bigger than 15. Chi =' + str(result.redchi) +
                      '. trying linear fit...')
        linear_result = fit_linear(x, y, make_plot=make_plot)
        if linear_result[1].redchi > result.redchi:
            raise ValueError('The reduced chi of the linear fit is even worse . Chi =' + str(linear_result[1].redchi) +
                             '. Chi of both fits is too large!. Try to begin the sweep from a later time point')
        else:
            result = linear_result
            best_function = 'linear'
    else:
        best_function = 'exponential'
    return best_function, result


def fit_biexponential(x, y, y0_1, y0_2, y_ss_1, y_ss_2, tau_1, tau_2,
                      make_plot=False):  # not yet fully implemented with guesser etc.
    fit_model = Model(two_first_oder_sys_responses)
    result = fit_model.fit(y, t=x, y0_1=y0_1, y0_2=y0_2, y_ss_1=y_ss_1, y_ss_2=y_ss_2, tau_1=tau_1, tau_2=tau_2)

    if make_plot:
        print(result.fit_report())
        plt.plot(x, y, 'bo')
        plt.plot(x, result.best_fit, 'r-', label='best fit')
        plt.legend(loc='best')
        plt.show()

    return result


def fit_pre_light(sweep, initial_fit_type, t0=None, make_plot=True):
    sweep_data = sweep.get_sweep_data()
    sweep_times = sweep_data['times']
    sweep_currents = sweep_data['currents']
    t_light_on = sweep_data['shutter on']
    recorded_t_light_on = get_closest_value_from_ordered_array(t_light_on, sweep_times)
    t_light_on_index = get_index_of_unique_value(recorded_t_light_on, sweep_times)
    if t0 is None:  # if starting time for fit is not specified, 1.5 s before the light will be taken
        t0 = t_light_on - 1.5
    assert (t0 > sweep_data['clamp on']), 'the first fit should not start before the capacitance peak: ' + str(
        t0) + ' > ' + str(sweep_data['clamp on'])
    assert sweep_times[0] <= t0 <= sweep_times[-1], 't0 is out of range sweep interval'
    recorded_t0 = get_closest_value_from_ordered_array(t0, sweep_times)
    t0_index = get_index_of_unique_value(recorded_t0, sweep_times)

    fit_time = sweep_times[t0_index:t_light_on_index]
    fit_current = sweep_currents[t0_index:t_light_on_index]
    if initial_fit_type == 'exponential':
        return fit_exponential(fit_time, fit_current, make_plot=make_plot)
    if initial_fit_type == 'linear':
        return fit_linear(fit_time, fit_current, make_plot=make_plot)
    else:
        raise NotImplementedError('this function was not implemented for the initial_fit_type:', initial_fit_type)


def calculate_linear_photocurrent_baseline(sweep, t_ss=None):
    sweep_data = sweep.get_sweep_data()
    sweep_times = sweep_data['times']
    sweep_currents = sweep_data['currents']
    t_light_on = sweep_data['shutter on']
    t_light_off = sweep_data['shutter off']
    t_clamp_off = sweep_data['clamp off']
    recorded_t_clamp_off = get_closest_value_from_ordered_array(t_clamp_off, sweep_times)
    t_clamp_off_index = get_index_of_unique_value(recorded_t_clamp_off, sweep_times)
    if t_ss is None:  # if starting time for fit is not specified, 1.5 s before the clamp is off will be taken
        t_ss = t_clamp_off - 0.5
    assert (
            t_ss > t_light_off), 'the steady state time point should not start before the light is off: {0} > {1}'.format(
        str(t_ss), str(t_light_off))
    assert sweep_times[0] <= t_ss <= sweep_times[-1], 'the steady state time is out of the range of the sweep interval'
    recorded_t_light_on = get_closest_value_from_ordered_array(t_light_on, sweep_times)
    t_light_on_index = get_index_of_unique_value(recorded_t_light_on, sweep_times)
    recorded_t_ss = get_closest_value_from_ordered_array(t_ss, sweep_times)
    t_ss_index = get_index_of_unique_value(recorded_t_ss, sweep_times)

    # before the light
    baseline = np.zeros(len(sweep_currents))

    # during the light and after
    deltay = np.average(sweep_currents[(t_ss_index - 5):(t_ss_index + 5)])
    deltax = recorded_t_ss - t_light_on
    slope = deltay / deltax

    t_value_index = 0
    for t_value in sweep_times:
        if recorded_t_light_on <= t_value <= recorded_t_ss:
            baseline[t_value_index] = linear(t_value-recorded_t_light_on, slope, 0)
        elif t_value > recorded_t_ss:
            baseline[t_value_index] = linear(recorded_t_ss-recorded_t_light_on, slope, 0)
        t_value_index += 1

    return baseline


def estimate_data_with_fit(t, function, fit_result, t_shift=0):
    fit_result_values = fit_result.best_values
    y = np.zeros(shape=t.shape)
    if function == 'exponential':
        y0 = fit_result_values['y0']
        y_ss = fit_result_values['y_ss']
        tau = fit_result_values['tau']
        for i in range(len(t)):
            y[i] = first_oder_sys_response(t[i], y0, y_ss, tau)
    if function == 'linear':
        m = fit_result_values['m']
        y0 = fit_result_values['y0']
        for i in range(len(t)):
            y[i] = linear(t[i], m, y0)
    return y
