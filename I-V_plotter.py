import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from _helpers import truncate
from _importer import import_sweeps_from_csv

### measurements of each construct ###
# first five columns of RQ and first four columns of RQ_construct7 are from the same cells (excluding Na10)!
measurement_names = {
    'RQ_pH7.5_Na': ['2019_11_26_0070', '2019_11_15_0061', '2019_11_12_0042', '2019_11_12_0022', '2019_11_11_0012'],
    'RQ_pH7.5_K': ['2019_11_26_0068', '2019_11_15_0048', '2019_11_12_0034', '2019_11_12_0017', '2019_11_11_0014','2019_11_15_0037'],
    'RQ_pH10_K': ['2019_11_26_0069', '2019_11_15_0057', '2019_11_12_0037', '2019_11_12_0021', '2019_11_11_0013','2019_11_15_0057'],
    'RQ_construct7_pH7.5_Na': ['2019_10_29_0031','2019_10_29_0063','2019_10_29_0085','2019_11_15_0098'],
    'RQ_construct7_pH10_Na': ['2019_10_29_0033','2019_11_15_0100'],
    'RQ_construct7_pH7.5_K': ['2019_10_29_0025','2019_10_29_0060','2019_10_29_0092','2019_11_15_0085','2019_10_29_0080','2019_10_29_0094','2019_11_13_0059'],
    'RQ_construct7_pH10_K': ['2019_10_29_0027','2019_10_29_0061','2019_10_29_0093','2019_11_15_0094','2019_10_29_0081','2019_10_29_0095','2019_11_13_0081']
}


### params ###
outputsFolder = '/Volumes/PENDISK/analysis_results/'
sweepsSuffix = '_sweeps.csv'

RefSweepsOutput = '/Volumes/PENDISK/analysis_results/2019_11_11_0012_sweeps.csv'
RefWhichVoltage = '0'
bestFitChiConstraints = {
    'min_red_chi_squared_improvement_ratio': 1.1,  # calculated as prev_val/curr_val
    'min_red_chi_squared_value': 0.95,
    'max_r_squared_deterioration_ratio': 1.33  # calculated as prev_val/curr_val
}
bestFitRConstraints = {
    'min_r_squared_delta_improvement': 0.01,  # calculated as prev_val/curr_val
}


### functions ###
def get_path_list(construct_name):  # name as appears under the dic "measurements"
    const_path_list = []
    for measurement in measurement_names[construct_name]:
        const_path_list.append(outputsFolder + measurement + sweepsSuffix)
    return {'name': construct_name, 'path list': const_path_list}


def normalize_measurement(measurement_dic, ref_measurement_dic, normalization_voltage=0):
    measurement_currents, measurement_currents_std = measurement_dic[
                                                         "currents"], measurement_dic["currents_std"]
    ref_measurement_voltages, ref_measurement_currents, ref_measurement_currents_std = ref_measurement_dic["voltages"], \
                                                                                       ref_measurement_dic[
                                                                                           "currents"], \
                                                                                       ref_measurement_dic[
                                                                                           "currents_std"]
    ref_measurement_best_fit_result = best_poly_fit(ref_measurement_voltages,
                                                    ref_measurement_currents)  # opt: y_SD=ref_measurement_dic["currents_std"]
    ref_fit_polynomial = ref_measurement_best_fit_result['polynomial']
    ref_current_at_norm_voltage = ref_fit_polynomial(normalization_voltage)
    normalized_currents = np.asarray(measurement_currents) / ref_current_at_norm_voltage
    normalization_normalized_currents_std = np.asarray(measurement_currents_std) / ref_current_at_norm_voltage
    result = {
        "name": measurement_dic["name"],
        "voltages": measurement_dic["voltages"],
        "currents": normalized_currents,
        "currents_std": normalization_normalized_currents_std,
        "voltages_std": measurement_dic["voltages_std"]
    }
    return result


def average_measurements(construct_paths_list):
    name = construct_paths_list['name']
    construct_paths_list = construct_paths_list['path list']
    assert len(construct_paths_list) >= 2, "At least 2 measurements for averaging!"
    voltages_list_of_lists = []
    currents_list_of_lists = []
    for measurement in construct_paths_list:
        measurement_data = import_sweeps_from_csv(measurement)
        measurement_voltages, measurement_currents = measurement_data["voltages"], measurement_data[
            "currents"]
        voltages_list_of_lists.append(measurement_voltages)
        currents_list_of_lists.append(measurement_currents)
    result = {
        'name': name,
        'voltages': np.mean(voltages_list_of_lists, axis=0),
        'voltages_std': np.std(voltages_list_of_lists, axis=0),
        'currents': np.mean(currents_list_of_lists, axis=0),
        'currents_std': np.std(currents_list_of_lists, axis=0)
    }
    return result


# Polynomial Regression
def polyfit_with_stats(x, y, degree, y_SD=None):
    results = {'degree': degree}

    if y_SD is not None:
        coeffs = np.polyfit(x, y, degree, w=1 / np.asanyarray(y_SD))  # Weights calculated as 1/std of currents
    else:
        coeffs = np.polyfit(x, y, degree)

    '''p, C_p = np.polyfit(x, y, degree, cov=True)  #

    # Do the interpolation for plotting:
    t = np.linspace(x[0], x[-1], 500)
    # Matrix with rows 1, t, t**2, ...:
    TT = np.vstack([t ** (degree - i) for i in range(degree + 1)]).T
    yi = np.dot(TT, p)  # matrix multiplication calculates the polynomial values
    C_yi = np.dot(TT, np.dot(C_p, TT.T))  # C_y = TT*C_z*TT.T
    sig_yi = np.sqrt(np.diag(C_yi))  # Standard deviations are sqrt of diagonal

    # Do the plotting:
    fg, ax = plt.subplots(1, 1)
    ax.set_title("Fit for Polynomial (degree {}) with $\pm1\sigma$-interval".format(degree))
    ax.fill_between(t, yi + sig_yi, yi - sig_yi, alpha=.25)
    ax.plot(t, yi, '-')
    ax.plot(x, y, 'ro')
    ax.axis('tight')

    fg.canvas.draw()
    plt.show()'''

    polynomial = np.poly1d(coeffs)

    results['coefficients'] = list(coeffs)
    results['polynomial'] = polynomial

    # following is based on Wikipedia: Coefficient_of_determination
    y = np.asanyarray(y)
    f = polynomial(x)  # vector, fit values for x_data
    y_bar = np.mean(y)
    ss_res = np.sum((y - f) ** 2)
    ss_tot = np.sum((y - y_bar) ** 2)

    r_squared = 1 - ss_res / ss_tot

    results['r-squared'] = r_squared

    # following is based in Wikipedia : Reduced_chi-squared_statistic
    if y_SD is not None:  # if to weigh chi-sq based on (!ONLY) the y_data standard deviation
        y_SD = np.asanyarray(y_SD)
        chi_squared = np.sum(((y - f) ** 2) / y_SD)
    else:
        chi_squared = np.sum((y - f) ** 2)

    reduced_chi_squared = chi_squared / (len(x) - (degree + 1))

    results['red-chi-squared'] = reduced_chi_squared

    return results


def best_poly_fit(x, y, y_SD=None, best_r=True):
    result = polyfit_with_stats(x, y, 1, y_SD=y_SD)
    for deg in range(2, 10):
        next_deg_result = polyfit_with_stats(x, y, deg, y_SD=y_SD)

        if best_r:
            r_squared_delta = next_deg_result['r-squared'] - result['r-squared']
            better_fit_condition = r_squared_delta > bestFitRConstraints['min_r_squared_delta_improvement']
        else:
            red_chi_sq_ratio = result['red-chi-squared'] / next_deg_result['red-chi-squared']
            r_squared_ratio = result['r-squared'] / next_deg_result['r-squared']
            better_fit_condition = red_chi_sq_ratio > bestFitChiConstraints['min_red_chi_squared_improvement_ratio'] and \
                                   next_deg_result['red-chi-squared'] > bestFitChiConstraints[
                                       'min_red_chi_squared_value'] and \
                                   r_squared_ratio < bestFitChiConstraints['max_r_squared_deterioration_ratio']
        if better_fit_condition:
            result = next_deg_result
        else:
            return result


def get_closest_value_to_data(list_of_values, data):
    center_of_data = np.mean(data)
    return min(list_of_values, key=lambda x: abs(x - center_of_data))


def plot_iv_curve(measurements):
    assert type(measurements) == tuple or type(measurements) == dict, 'bad type! Is a {}'.format(type(measurements))
    fig, ax = plt.subplots(1)
    colorindex = 0
    colors = ['black', 'r', 'b', 'g', 'm', 'y']

    def plot_before_showing(measurement):
        name, voltages, currents, currents_std, voltages_std = measurement['name'], \
                                                               measurement['voltages'], \
                                                               measurement['currents'], \
                                                               measurement['currents_std'], \
                                                               measurement['voltages_std']

        bestFitResult = best_poly_fit(voltages, currents, y_SD=currents_std)
        # bestFitResult = polyfit_with_stats(voltagesList, currentsList, 2, y_SD=None)
        refPolyFunction = bestFitResult['polynomial']

        E_rev = get_closest_value_to_data(refPolyFunction.roots, voltages)

        bestFitDegree = bestFitResult['degree']
        bestFitRedChiSq = bestFitResult['red-chi-squared']
        bestFitRSquared = bestFitResult['r-squared']

        ax.errorbar(voltages, currents, yerr=currents_std, xerr=voltages_std, color=colors[colorindex],
                    linestyle='None', marker='x', capsize=5, capthick=1, ecolor=colors[colorindex], label=name)
        ax.plot(voltages, refPolyFunction(voltages), colors[colorindex],
                label='Polynomial fit (deg = {}) \n E_rev = {} mV'.format(bestFitDegree, str(truncate(E_rev, 2))))
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        handles, labels = ax.get_legend_handles_labels()
        # handles.append(mpatches.Patch(color='none', label='Red_chi^2 = ' + str(truncate(bestFitRedChiSq, 2))))
        # handles.append(mpatches.Patch(color='none', label='R^2 = ' + str(truncate(bestFitRSquared, 2))))
        # handles.append(mpatches.Patch(color='none', label='E_rev = {} mV'.format(str(truncate(E_rev, 2)))))
        ax.legend(handles=handles,fontsize="small")
        ax.set_xlabel('Holding Potential [mV]')
        # ax.set_xlim([-105, 80])
        #ax.set_ylim([-10, 110])
        ax.xaxis.set_label_coords(0.95, 0.38)
        ax.yaxis.set_label_coords(0.49, 0.85)
        ax.set_ylabel('Normalized \nPhotocurrents')

    if type(measurements) == tuple:
        for measurement in measurements:
            plot_before_showing(measurement)
            colorindex += 1

    elif type(measurements) == dict:
        plot_before_showing(measurements)

    else:
        raise ValueError(measurements)

    plt.savefig("/Volumes/PENDISK/figuresoutput/figure"+str(colorindex)+".pdf")
    plt.show()



### main ###



RQ_7_5_Na_averages = average_measurements(get_path_list('RQ_pH7.5_Na'))
RQ_7_5_Na_normalized_to_self = normalize_measurement(RQ_7_5_Na_averages, RQ_7_5_Na_averages)

RQ_7_5_K_averages = average_measurements(get_path_list('RQ_pH7.5_K'))
RQ_7_5_K_normalized_to_self = normalize_measurement(RQ_7_5_K_averages, RQ_7_5_K_averages)
RQ_7_5_K_normalized_to_7_5_Na = normalize_measurement(RQ_7_5_K_averages, RQ_7_5_Na_averages)

RQ_10_K_averages = average_measurements(get_path_list('RQ_pH10_K'))
RQ_10_K_normalized_to_self = normalize_measurement(RQ_10_K_averages, RQ_10_K_averages)
RQ_10_K_normalized_to_7_5_K = normalize_measurement(RQ_10_K_averages, RQ_7_5_K_averages)
RQ_10_K_normalized_to_7_5_Na = normalize_measurement(RQ_10_K_averages, RQ_7_5_Na_averages)

RQ_construct7_7_5_Na_averages = average_measurements(get_path_list('RQ_construct7_pH7.5_Na'))
RQ_construct7_7_5_Na_normalized_to_self = normalize_measurement(RQ_construct7_7_5_Na_averages,RQ_construct7_7_5_Na_averages)
RQ_construct7_7_5_Na_normalized_to_RQ_7_5_Na = normalize_measurement(RQ_construct7_7_5_Na_averages, RQ_7_5_Na_averages)

RQ_construct7_10_Na_averages = average_measurements(get_path_list('RQ_construct7_pH10_Na'))
RQ_construct7_10_Na_normalized_to_RQ_7_5_Na = normalize_measurement(RQ_construct7_10_Na_averages, RQ_7_5_Na_averages)
RQ_construct7_10_Na_normalized_to_RQ_construct7_7_5_Na = normalize_measurement(RQ_construct7_10_Na_averages, RQ_construct7_7_5_Na_averages)

RQ_construct7_7_5_K_averages = average_measurements(get_path_list('RQ_construct7_pH7.5_K'))
RQ_construct7_7_5_K_normalized_to_RQ_7_5_Na = normalize_measurement(RQ_construct7_7_5_K_averages, RQ_7_5_Na_averages)
RQ_construct7_7_5_K_normalized_to_RQ_7_5_K = normalize_measurement(RQ_construct7_7_5_K_averages,RQ_7_5_K_averages)
RQ_construct7_7_5_K_normalized_to_RQ_construct7_7_5_Na = normalize_measurement(RQ_construct7_7_5_K_averages, RQ_construct7_7_5_Na_averages)


RQ_construct7_10_K_averages = average_measurements(get_path_list('RQ_construct7_pH10_K'))
RQ_construct7_10_K_normalized_RQ_7_5_Na = normalize_measurement(RQ_construct7_10_K_averages, RQ_7_5_Na_averages)
RQ_construct7_10_K_normalized_RQ_10_K = normalize_measurement(RQ_construct7_10_K_averages,RQ_10_K_averages)
RQ_construct7_10_K_normalized_RQ_construct7_7_5_Na = normalize_measurement(RQ_construct7_10_K_averages, RQ_construct7_7_5_Na_averages)

RQ_measurements_ion_comparison = RQ_7_5_Na_normalized_to_self, RQ_7_5_K_normalized_to_7_5_Na, RQ_10_K_normalized_to_7_5_Na
RQ_construct7_measurements_ion_comparison = RQ_construct7_7_5_Na_normalized_to_self,RQ_construct7_7_5_K_normalized_to_RQ_construct7_7_5_Na,RQ_construct7_10_K_normalized_RQ_construct7_7_5_Na
rel_Na_measurements_construct_comparison = RQ_7_5_Na_normalized_to_self, RQ_construct7_7_5_Na_normalized_to_RQ_7_5_Na
rel_K_7_5_measurements_construct_comparison = RQ_7_5_K_normalized_to_self, RQ_construct7_7_5_K_normalized_to_RQ_7_5_K
rel_K_10_measurements_construct_comparison = RQ_10_K_normalized_to_self, RQ_construct7_10_K_normalized_RQ_10_K


#plot_iv_curve(import_sweeps_from_csv(outputsFolder+'2019_10_29_0013'+sweepsSuffix))
plot_iv_curve(rel_Na_measurements_construct_comparison)
plot_iv_curve(RQ_measurements_ion_comparison)
plot_iv_curve(RQ_construct7_measurements_ion_comparison)
plot_iv_curve(rel_K_7_5_measurements_construct_comparison)
plot_iv_curve(rel_K_10_measurements_construct_comparison)



