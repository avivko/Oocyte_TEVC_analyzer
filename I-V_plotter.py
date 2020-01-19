import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from _helpers import truncate
from _importer import import_sweeps_from_csv

### measurements of each construct ###
measurements = {
    'RQ_pH7.5_Na': ['2019_11_26_0070', '2019_11_15_0061', '2019_11_12_0042','2019_11_12_0022' ,'2019_11_11_0012'],
    'RQ_pH7.5_K': ['2019_11_26_0068', '2019_11_15_0048', '2019_11_12_0034', '2019_11_12_0017' ,'2019_11_11_0014'], #'2019_11_26_0063',
    'RQ_pH10_K': ['2019_11_26_0069', '2019_11_15_0057', '2019_11_12_0037', '2019_11_12_0021' ,'2019_11_11_0013'],


}

### params ###
outputsFolder = '/Volumes/PENDISK/analysis_results/'
sweepsSuffix = '_sweeps.csv'

RefSweepsOutput = '/Volumes/PENDISK/analysis_results/2019_11_12_0037_sweeps.csv'
RefWhichVoltage = '0'
bestFitConstraints = {
    'min_red_chi_squared_improvement_ratio': 1.1,  # calculated as prev_val/curr_val
    'min_red_chi_squared_value': 0.95,
    'max_r_squared_deterioration_ratio': 1.33  # calculated as prev_val/curr_val
}


### functions ###
def get_path_list(construct_name):  # name as appears under the dic "measurements"
    const_path_list = []
    for measurement in measurements[construct_name]:
        const_path_list.append(outputsFolder + measurement + sweepsSuffix)
    return const_path_list


def average_measurements(construct_paths_list):
    assert len(construct_paths_list) >= 2, "At least 2 measurements for averaging!"
    voltages_list_of_lists = []
    currents_list_of_lists = []
    for measurement in construct_paths_list:
        measurement_data = import_sweeps_from_csv(measurement)
        measurement_voltages, measurement_currents = measurement_data["voltagesAsList"], measurement_data[
            "currentsAsList"]
        voltages_list_of_lists.append(measurement_voltages)
        currents_list_of_lists.append(measurement_currents)
    result = {
        'average_voltages': np.mean(voltages_list_of_lists, axis=0),
        'std_voltages': np.std(voltages_list_of_lists, axis=0),
        'average_currents': np.mean(currents_list_of_lists, axis=0),
        'std_currents': np.std(currents_list_of_lists, axis=0)
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


def best_poly_fit(x, y, y_SD=None):
    result = polyfit_with_stats(x, y, 1, y_SD=y_SD)
    for deg in range(2, 10):
        next_deg_result = polyfit_with_stats(x, y, deg, y_SD=y_SD)
        red_chi_sq_ratio = result['red-chi-squared'] / next_deg_result['red-chi-squared']
        r_squared_ratio = result['r-squared'] / next_deg_result['r-squared']
        print(red_chi_sq_ratio,next_deg_result['red-chi-squared'],r_squared_ratio)
        better_fit_condition = red_chi_sq_ratio > bestFitConstraints['min_red_chi_squared_improvement_ratio'] and \
                next_deg_result['red-chi-squared'] > bestFitConstraints['min_red_chi_squared_value'] and \
                r_squared_ratio < bestFitConstraints['max_r_squared_deterioration_ratio']
        if better_fit_condition:
            result = next_deg_result
        else:
            return result


def get_closest_value_to_data(list_of_values, data):
    center_of_data = np.mean(data)
    return min(list_of_values, key=lambda x: abs(x - center_of_data))


### main ###
'''
RQ_7_5_Na_path_list = get_path_list('RQ_pH10_K')
RQ_7_5_Na_averages = average_measurements(RQ_7_5_Na_path_list)


voltagesList, currentsList, currentsSDList, voltagesSDList = RQ_7_5_Na_averages['average_voltages'], \
                                             RQ_7_5_Na_averages['average_currents'], \
                                             RQ_7_5_Na_averages['std_currents'],\
                                             RQ_7_5_Na_averages['std_voltages']
'''
refData = import_sweeps_from_csv(RefSweepsOutput)
voltagesList, currentsList, currentsSDList, voltagesSDList = refData["voltagesAsList"], \
                                                             refData["currentsAsList"],\
                                                             refData["currentsSDAsList"],\
                                                             refData["voltagesSDAsList"]


bestFitResult = best_poly_fit(voltagesList, currentsList)
refPolyFunction = bestFitResult['polynomial']
print(refPolyFunction.roots)
I_0 = refPolyFunction(0)
E_rev = get_closest_value_to_data(refPolyFunction.roots, voltagesList)
print(E_rev)
bestFitDegree = bestFitResult['degree']
bestFitRedChiSq = bestFitResult['red-chi-squared']
bestFitRSquared = bestFitResult['r-squared']

fig, ax = plt.subplots(1)
ax.plot(voltagesList, refPolyFunction(voltagesList), 'r',
        label='Polynomial fit (Degree={})'.format(bestFitDegree))
ax.errorbar(voltagesList, currentsList, yerr=currentsSDList, xerr=voltagesSDList, color='black', linestyle='None', marker='x', capsize=5, capthick=1, ecolor='black',label='Data point')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
handles, labels = ax.get_legend_handles_labels()
handles.append(mpatches.Patch(color='none', label='Red_chi^2 = ' + str(truncate(bestFitRedChiSq, 2))))
handles.append(mpatches.Patch(color='none', label='R^2 = ' + str(truncate(bestFitRSquared, 2))))
handles.append(mpatches.Patch(color='none', label='I(0mV) = {} nA'.format(str(truncate(I_0, 2)))))
handles.append(mpatches.Patch(color='none', label='E_rev = {} mV'.format(str(truncate(E_rev, 2)))))
ax.legend(handles=handles)
ax.set_xlabel('Voltage [mV]')
#ax.set_xlim([-105, 80])
#ax.set_ylim([-350, 350])
ax.xaxis.set_label_coords(0.95, 0.44)
ax.yaxis.set_label_coords(0.49, 0.85)
ax.set_ylabel('Steady-State \nPhotocurrents [nA]')
fig.show()
