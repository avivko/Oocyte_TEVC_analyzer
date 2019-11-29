import matplotlib.pyplot as plt
import pyabf as pyabf
import numpy as np
from fitting import *


class ActiveAbf:
    def __init__(self, abf_file):
        self.abf_data = pyabf.ABF(abf_file)
        self._abf_file_path = abf_file
        self._data_points_per_sec = self.abf_data.dataRate
        self._sweeps = self.abf_data.sweepCount

    def sweep_count(self):
        return self.abf_data.sweepCount

    def which_abf_file(self):
        return self._abf_file_path

    def get_sweep_voltages(self):
        sweep_voltages = {}
        for sweepNumber in range(self._sweeps):
            self.abf_data.setSweep(sweepNumber)
            sweep_voltages[sweepNumber] = self.abf_data.sweepC[round(len(self.abf_data.sweepC) / 2)]
        return sweep_voltages

    def get_raw_abf_data(self):
        some_data = {}
        for sweepNumber in range(self._sweeps):
            self.abf_data.setSweep(sweepNumber)
            some_data[sweepNumber] = {
                'sweep number': sweepNumber,
                'sweep currents (ADC)': self.abf_data.sweepY,
                'sweep input voltages (DAC)': self.abf_data.sweepC,
                'sweep times (seconds)': self.abf_data.sweepX
            }
        return some_data


class sweep(ActiveAbf):
    def __init__(self, abf_file, sweep_nr):
        super().__init__(abf_file)
        self.abf_data.setSweep(sweep_nr)
        self.t_clamp_on = self.abf_data.sweepEpochs.p1s[1] * self.abf_data.dataSecPerPoint
        self.t_shutter_on = self.abf_data.sweepEpochs.p1s[2] * self.abf_data.dataSecPerPoint
        self.t_shutter_off = self.abf_data.sweepEpochs.p1s[3] * self.abf_data.dataSecPerPoint
        self.t_clamp_off = self.abf_data.sweepEpochs.p1s[4] * self.abf_data.dataSecPerPoint
        self.currents = self.abf_data.sweepY
        self.currents_title = self.abf_data.sweepLabelY
        self.times = self.abf_data.sweepX
        self.times_title = self.abf_data.sweepLabelX
        self.input_voltage = self.abf_data.sweepC
        self.input_voltage_title = 'Digital Input Clamp Voltage (mV)'
        self.abf_data.setSweep(sweep_nr, 1)
        self.voltages = self.abf_data.sweepY
        self.voltages_title = self.abf_data.sweepLabelY
        self.abf_data.setSweep(sweep_nr, 2)
        self.shutter = self.abf_data.sweepY
        self.shutter_title = 'Shutter Voltage (V)'

    def get_sweep_data(self):
        return {
            'currents': self.currents,
            'currents title': self.currents_title,
            'times': self.times,
            'times title': self.times_title,
            'voltages': self.voltages,
            'voltages title': self.voltages_title,
            'shutter': self.shutter,
            'shutter title': self.shutter_title,
            'shutter on': self.t_shutter_on,
            'shutter off': self.t_shutter_off,
            'clamp on': self.t_clamp_on,
            'clamp off': self.t_clamp_off,
            'input clamp voltage': self.input_voltage,
            'input clamp voltage title': self.input_voltage_title
        }

    def set_corrected_currents(self, corrected_currents):
        assert corrected_currents.shape == self.currents.shape, 'new currents do not have the same shape as the ' \
                                                                 'previous ones '
        self.currents = corrected_currents


def correct_current_via_pre_light_fit(sweep, initial_function='exponential'):
    sweep_times = sweep.times
    sweep_currents = sweep.currents
    best_function, pre_light_fit_result = fit_pre_light(sweep, initial_function, make_plot=False)
    pre_light_fit_baseline = estimate_data_with_fit(sweep_times, best_function, pre_light_fit_result)
    baseline_corrected_currents = sweep_currents - pre_light_fit_baseline
    return baseline_corrected_currents


def correct_current_via_linear_baseline(sweep, initial_function_pre_light='exponential'):
    sweep_times = sweep.times
    sweep_currents = sweep.currents
    pre_light_best_function, pre_light_fit_result = fit_pre_light(sweep, initial_function_pre_light, make_plot=False)
    pre_light_fit_baseline = estimate_data_with_fit(sweep_times, pre_light_best_function, pre_light_fit_result)
    pre_light_baseline_corrected_currents = sweep_currents - pre_light_fit_baseline
    sweep.set_corrected_currents(pre_light_baseline_corrected_currents)
    linear_light_baseline = calculate_linear_photocurrent_baseline(sweep)
    sweep_currents = sweep.currents
    baseline_corrected_currents = sweep_currents - linear_light_baseline
    return baseline_corrected_currents


def auto_interval_to_plot(sweep):
    t_start = sweep.t_shutter_on-1
    t_end = sweep.t_shutter_off+1
    if (t_start < sweep.t_clamp_on) or (t_end > sweep.t_clamp_off):
        t_start = sweep.t_shutter_on - 0.5
        t_end = sweep.t_shutter_off + 0.5

    first_element = get_index_of_closest_value(t_start, sweep.times)
    last_element = get_index_of_closest_value(t_end, sweep.times)
    return first_element, last_element


def plot_sweep(sweep, plot_interval=None, corrected=False):
    if plot_interval is None:
        plot_interval = auto_interval_to_plot(sweep)
    else:
        assert (type(plot_interval) == list and len(plot_interval) == 2)
    if not corrected:
        sweep_time = sweep.times
        sweep_current = sweep.currents
        sweep_voltage = sweep.voltages
    elif corrected == 'pre_light_only':
        sweep_time = sweep.times
        sweep_current = correct_current_via_pre_light_fit(sweep)
        sweep_voltage = sweep.voltages
    elif corrected == 'pre_and_after_light':
        sweep_time = sweep.times
        sweep_current = correct_current_via_linear_baseline(sweep)
        sweep_voltage = sweep.voltages
    else:
        raise ValueError('corrected should be bool: False / pre_light_only / pre_and_after_light. Is, however, ', corrected)
    fig, axs = plt.subplots(2)
    axs[0].plot(sweep_time[plot_interval[0]:plot_interval[1]], sweep_current[plot_interval[0]:plot_interval[1]])
    axs[0].set(xlabel=sweep.times_title, ylabel=sweep.currents_title)
    axs[1].plot(sweep_time[plot_interval[0]:plot_interval[1]], sweep_voltage[plot_interval[0]:plot_interval[1]])
    axs[1].set(xlabel=sweep.times_title, ylabel=sweep.voltages_title)

    for ax in axs.flat:
        ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.grid(alpha=.2)
        ax.axvspan(sweep.t_shutter_on, sweep.t_shutter_off, color='orange', alpha=.3, lw=0)

    plt.show()


def get_voltage_changes(ActiveAbf):
    avg_voltages_and_their_changes = {}
    nr_of_sweeps = ActiveAbf.sweep_count()
    for i in range(nr_of_sweeps):
        avg_sweep_voltages_and_changes = {}
        sweep_number = nr_of_sweeps - 1 - i
        sweep_interation = sweep(ActiveAbf.which_abf_file(), sweep_number)
        sweep_times = sweep_interation.times
        sweep_voltages = sweep_interation.voltages

        t_light_on = sweep_interation.t_shutter_on
        t_t_light_on_index = get_index_of_closest_value(t_light_on, sweep_times)
        avg_voltage_before_light_at_ss = np.average(sweep_voltages[t_t_light_on_index-10:t_t_light_on_index])
        avg_sweep_voltages_and_changes['before (at ss)'] = avg_voltage_before_light_at_ss

        t_light_off = sweep_interation.t_shutter_off
        t_light_off_index = get_index_of_closest_value(t_light_off, sweep_times)
        avg_voltage_during_light_at_ss = np.average(sweep_voltages[t_light_off_index - 10:t_light_off_index])
        avg_sweep_voltages_and_changes['during (at ss)'] = avg_voltage_during_light_at_ss

        t_clamp_off = sweep_interation.t_clamp_off
        t_clamp_off_index = get_index_of_closest_value(t_clamp_off, sweep_times)
        avg_voltage_after_light_at_ss = np.average(sweep_voltages[t_clamp_off_index - 10:t_clamp_off_index])
        avg_sweep_voltages_and_changes['after (at ss)'] = avg_voltage_after_light_at_ss

        delta_before_and_during_light = abs(avg_voltage_during_light_at_ss-avg_voltage_before_light_at_ss)
        avg_sweep_voltages_and_changes['delta(before,during)'] = delta_before_and_during_light
        delta_before_and_after_light = abs(avg_voltage_after_light_at_ss-avg_voltage_before_light_at_ss)
        avg_sweep_voltages_and_changes['delta(before,after)'] = delta_before_and_after_light
        avg_voltages_and_their_changes['sweep'+str(sweep_number)] = avg_sweep_voltages_and_changes

    return avg_voltages_and_their_changes


def plot_all_sweeps(ActiveAbf, plot_interval=None, corrected=False):
    if plot_interval is None:
        first_sweep = sweep(ActiveAbf.which_abf_file(), 1)
        plot_interval = auto_interval_to_plot(first_sweep)
    else:
        assert (type(plot_interval) == list and len(plot_interval) == 2)
    nr_of_sweeps = ActiveAbf.sweep_count()
    fig, ax = plt.subplots(1)
    for i in range(nr_of_sweeps):
        sweep_number = nr_of_sweeps - 1 - i
        sweep_interation = sweep(ActiveAbf.which_abf_file(), sweep_number)
        if not corrected:
            time = sweep_interation.times
            current = sweep_interation.currents
            voltage = sweep_interation.voltages
        elif corrected == 'pre_light_only':
            time = sweep_interation.times
            current = correct_current_via_pre_light_fit(sweep_interation)
            voltage = sweep_interation.voltages
        elif corrected == 'pre_and_after_light':
            time =  sweep_interation.times
            current = correct_current_via_linear_baseline(sweep_interation)
            voltage = sweep_interation.voltages
        else:
            raise ValueError('corrected should be bool: True / False . Is, however,', type(corrected))
        ax.plot(time[plot_interval[0]:plot_interval[1]], current[plot_interval[0]:plot_interval[1]], alpha=.5,
                    label="{} mV".format(sweep_interation.input_voltage[round(len(
                        sweep_interation.input_voltage) / 2)]))
        ax.legend(loc='upper left', prop={'size': 8})
        if sweep_number == 0:
            ax.set(xlabel=sweep_interation.times_title, ylabel=sweep_interation.currents_title)
            ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
            ax.grid(alpha=.2)
            ax.axvspan(sweep_interation.t_shutter_on, sweep_interation.t_shutter_off, color='orange', alpha=.3, lw=0)
    plt.show()
