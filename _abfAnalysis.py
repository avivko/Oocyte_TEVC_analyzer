import matplotlib.pyplot as plt
import pyabf as pyabf
import numpy as np
import os
from pathlib import Path
from pandas import DataFrame
from _fitting import *
import logging

### parameters ###
photocurrents_ss_duration = 0.5  # [sec] , is the duration of time before the shutter is closed
                                    # where steadystate photucurrents are assumed
plotting_buffer = 0.5            # [sec] , is the duration of time before and after the photocurrents that will be shown in the
                                    # plot

##################

class ActiveAbf:
    def __init__(self, abf_file):
        self.abf_data = pyabf.ABF(abf_file)
        self._abf_file_path = abf_file
        self._data_points_per_sec = self.abf_data.dataRate
        self._nr_of_sweeps = self.abf_data.sweepCount
        self.sweep_list = {}

    def sweep_count(self):
        return self.abf_data.sweepCount

    def which_abf_file(self):
        return self._abf_file_path

    def get_sweep_input_voltages(self):
        sweep_voltages = {}
        for sweepNumber in range(self._nr_of_sweeps):
            self.abf_data.setSweep(sweepNumber)
            sweep_voltages[sweepNumber] = self.abf_data.sweepC[round(len(self.abf_data.sweepC) / 2)]
        return sweep_voltages

    def get_voltage_changes(self):
        avg_voltages_and_their_changes = {}
        nr_of_sweeps = self.sweep_count()
        for i in range(nr_of_sweeps):
            avg_sweep_voltages_and_changes = {}
            sweep_number = nr_of_sweeps - 1 - i
            sweep_interation = self.get_sweep(sweep_number)
            sweep_times = sweep_interation.times
            sweep_voltages = sweep_interation.voltages

            t_light_on = sweep_interation.t_shutter_on
            t_t_light_on_index = get_index_of_closest_value(t_light_on, sweep_times)
            avg_voltage_before_light_at_ss = np.average(sweep_voltages[t_t_light_on_index - 10:t_t_light_on_index])
            avg_sweep_voltages_and_changes['before (at ss)'] = avg_voltage_before_light_at_ss

            t_light_off = sweep_interation.t_shutter_off
            t_light_off_index = get_index_of_closest_value(t_light_off, sweep_times)
            t_stst_start = t_light_off - photocurrents_ss_duration
            t_stst_start_index = get_index_of_closest_value(t_stst_start, sweep_times)
            avg_voltage_during_light_at_ss = np.average(sweep_voltages[t_stst_start_index:t_light_off_index])
            voltage_sd_during_light_at_ss = np.std(sweep_voltages[t_stst_start_index:t_light_off_index])
            avg_sweep_voltages_and_changes['during (at ss)'] = avg_voltage_during_light_at_ss
            avg_sweep_voltages_and_changes['sd of during (at ss)'] = voltage_sd_during_light_at_ss

            t_clamp_off = sweep_interation.t_clamp_off
            t_clamp_off_index = get_index_of_closest_value(t_clamp_off, sweep_times)
            avg_voltage_after_light_at_ss = np.average(sweep_voltages[t_clamp_off_index - 10:t_clamp_off_index])
            avg_sweep_voltages_and_changes['after (at ss)'] = avg_voltage_after_light_at_ss

            delta_before_and_during_light = abs(avg_voltage_during_light_at_ss - avg_voltage_before_light_at_ss)
            delta_after_and_during_light = abs(avg_voltage_during_light_at_ss - avg_voltage_after_light_at_ss)
            avg_sweep_voltages_and_changes['voltage jump'] = abs(
                delta_before_and_during_light + delta_after_and_during_light) / 2
            delta_before_and_after_light = abs(avg_voltage_after_light_at_ss - avg_voltage_before_light_at_ss)
            avg_sweep_voltages_and_changes['voltage drift'] = delta_before_and_after_light
            avg_voltages_and_their_changes['sweep' + str(sweep_number)] = avg_sweep_voltages_and_changes

        return avg_voltages_and_their_changes

    def get_stst_currents(self):
        stst_currents = {}
        nr_of_sweeps = self.sweep_count()
        for i in range(nr_of_sweeps):
            sweep_number = nr_of_sweeps - 1 - i
            sweep_interation = self.get_sweep(sweep_number)
            assert sweep_interation.currents_are_corrected, "currents are not yet corrected! could not get steady" \
                                                            " states currents before correction"
            sweep_times = sweep_interation.times
            sweep_currents = sweep_interation.currents
            sweep_t_light_off = sweep_interation.t_shutter_off
            t_stst_end = sweep_t_light_off
            t_stst_end_index = get_index_of_closest_value(t_stst_end, sweep_times)
            t_stst_start = sweep_t_light_off - photocurrents_ss_duration
            t_stst_start_index = get_index_of_closest_value(t_stst_start, sweep_times)
            stst_current = np.average(sweep_currents[t_stst_start_index:t_stst_end_index])
            stst_current_sd = np.std(sweep_currents[t_stst_start_index:t_stst_end_index])
            stst_currents['sweep' + str(sweep_number)] = {'ss current': stst_current,
                                                          'ss current sd': stst_current_sd}
        return stst_currents

    def get_raw_abf_data(self):
        some_data = {}
        for sweepNumber in range(self._nr_of_sweeps):
            self.abf_data.setSweep(sweepNumber)
            some_data[sweepNumber] = {
                'sweep number': sweepNumber,
                'sweep currents (ADC)': self.abf_data.sweepY,
                'sweep input voltages (DAC)': self.abf_data.sweepC,
                'sweep times (seconds)': self.abf_data.sweepX
            }
        return some_data

    def get_sweep(self, sweep_num):
        try:
            return self.sweep_list['sweep' + str(sweep_num)]
        except KeyError:
            return self.create_sweep_obj(sweep_num)

    def create_sweep_obj(self, sweep_num):
        returned_sweep = sweep(self._abf_file_path, sweep_num)
        self.sweep_list['sweep' + str(sweep_num)] = returned_sweep
        return self.sweep_list['sweep' + str(sweep_num)]

    def make_output_folder(self):
        analysis_results_folder = Path(str(Path(self.which_abf_file()).parent) + '/analysis_results/')
        Path.mkdir(analysis_results_folder, exist_ok=True)
        return analysis_results_folder

    def export_analyzed_abf_data_to_csv(self):
        name_of_abf = Path(self.which_abf_file()).stem
        currents_data = {"00_sweep_time_point[sec]": self.sweep_list["sweep0"].times}
        for i in range(self._nr_of_sweeps):
            sweep_in_abf = self.get_sweep(i)
            col_index_uncorrected = 1+2*i
            col_index_corrected = 2+2*i
            if col_index_corrected < 10:
                pre_index = '0'
            else:
                pre_index = ''
            currents_data[pre_index+str(col_index_uncorrected)+"_sweep" + str(i) + "_uncorrected_current[nA]"] = sweep_in_abf.original_currents
            currents_data[pre_index+str(col_index_corrected)+"_sweep" + str(i) + "_corrected_current[nA]"] = sweep_in_abf.currents

        currents_df = DataFrame(currents_data, columns=sorted(currents_data.keys()))
        output_folder = self.make_output_folder()
        currents_df.to_csv(str(output_folder) + '/' + str(name_of_abf) + '_currents.csv', index=None, header=True)

        stst_currents = self.get_stst_currents()
        voltage_changes = self.get_voltage_changes()
        sweeps_data = {"0_sweep_nr": np.array([str(i) for i in range(self._nr_of_sweeps)]),
                       "1_input_voltage[mV]": np.array(
                           [self.get_sweep(i).input_voltage for i in range(self._nr_of_sweeps)]),
                       "2_currents_during_light_at_steadystate[nA]": np.array(
                           [stst_currents["sweep" + str(i)]['ss current'] for i in range(self._nr_of_sweeps)]),
                       "3_SD_of_currents_during_light_at_steadystate[nA]": np.array(
                           [stst_currents["sweep" + str(i)]['ss current sd'] for i in range(self._nr_of_sweeps)]),
                       "4_voltage_during_light_at_steadystate[mV]": np.array(
                           [voltage_changes["sweep" + str(i)]['during (at ss)'] for i in range(self._nr_of_sweeps)]),
                       "5_SD_of_voltage_during_light_at_steadystate[mV]": np.array(
                           [voltage_changes["sweep" + str(i)]['sd of during (at ss)'] for i in range(self._nr_of_sweeps)]),
                       "6_voltage_jump[mV]": np.array(
                           [voltage_changes["sweep" + str(i)]['voltage jump'] for i in range(self._nr_of_sweeps)]),
                       "7_voltage_drift[mV]": np.array(
                           [voltage_changes["sweep" + str(i)]['voltage drift'] for i in range(self._nr_of_sweeps)])}
        sweeps_df = DataFrame(sweeps_data, columns=sorted(sweeps_data.keys()))

        sweeps_df.to_csv(str(output_folder) + '/' + str(name_of_abf) + '_sweeps.csv', index=None, header=True)


class sweep(ActiveAbf):
    def __init__(self, abf_file, sweep_nr):
        super().__init__(abf_file)
        self.abf_data.setSweep(sweep_nr)
        self.sweep_nr = sweep_nr
        self.t_clamp_on = self.abf_data.sweepEpochs.p1s[1] * self.abf_data.dataSecPerPoint
        self.t_shutter_on = self.abf_data.sweepEpochs.p1s[2] * self.abf_data.dataSecPerPoint
        self.t_shutter_off = self.abf_data.sweepEpochs.p1s[3] * self.abf_data.dataSecPerPoint
        self.t_clamp_off = self.abf_data.sweepEpochs.p1s[4] * self.abf_data.dataSecPerPoint
        self.original_currents = self.abf_data.sweepY
        self.currents_are_corrected = False
        self.correction_type = None
        self.currents = self.abf_data.sweepY
        self.currents_title = self.abf_data.sweepLabelY
        self.times = self.abf_data.sweepX
        self.times_title = self.abf_data.sweepLabelX
        self.input_voltage = self.abf_data.sweepC[round(len(self.abf_data.sweepC) / 2)]
        self.input_voltage_title = 'Digital Input Clamp Voltage (mV)'
        self.abf_data.setSweep(sweep_nr, 1)
        self.voltages = self.abf_data.sweepY
        self.voltages_title = self.abf_data.sweepLabelY
        self.abf_data.setSweep(sweep_nr, 2)
        self.shutter = self.abf_data.sweepY
        self.shutter_title = 'Shutter Voltage (V)'

    def get_sweep_data(self):
        return {
            'sweep nr': self.sweep_nr,
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

    def set_corrected_currents(self, corrected_currents, correction_type):
        assert corrected_currents.shape == self.currents.shape, 'new currents do not have the same shape as the ' \
                                                                'previous ones '
        self.currents = corrected_currents
        self.currents_are_corrected = True
        self.correction_type = correction_type


def correct_current_via_pre_light_fit(sweep, initial_function='exponential'):
    sweep_times = sweep.times
    sweep_currents = sweep.currents
    best_function, pre_light_fit_result = fit_pre_light(sweep, initial_function)
    pre_light_fit_baseline = estimate_data_with_fit(sweep_times, best_function, pre_light_fit_result)
    baseline_corrected_currents = sweep_currents - pre_light_fit_baseline
    sweep.set_corrected_currents(baseline_corrected_currents, 'pre_light_only')
    return baseline_corrected_currents


def correct_current_via_linear_baseline(sweep, initial_function_pre_light='exponential',
                                        initial_function_after_light='exponential'):
    correct_current_via_pre_light_fit(sweep, initial_function=initial_function_pre_light)
    linear_light_baseline = calculate_linear_photocurrent_baseline(sweep,
                                                                   fit_after_function=initial_function_after_light)
    sweep_currents = sweep.currents
    baseline_corrected_currents = sweep_currents - linear_light_baseline
    sweep.set_corrected_currents(baseline_corrected_currents, 'pre_and_after_light')
    return baseline_corrected_currents


def auto_interval_to_plot(sweep):
    t_start = sweep.t_shutter_on - plotting_buffer
    t_end = sweep.t_shutter_off + plotting_buffer
    if (t_start < sweep.t_clamp_on) or (t_end > sweep.t_clamp_off):
        t_start = sweep.t_shutter_on - plotting_buffer/2
        t_end = sweep.t_shutter_off + plotting_buffer/2

    first_element = get_index_of_closest_value(t_start, sweep.times)
    last_element = get_index_of_closest_value(t_end, sweep.times)
    return first_element, last_element


def correct_currents(sweep, correction):
    if correction == 'pre_light_only':
        corrected_currents = correct_current_via_pre_light_fit(sweep)
    elif correction == 'pre_and_after_light':
        corrected_currents = correct_current_via_linear_baseline(sweep)
    else:
        logging.error('corrected should pre_light_only / pre_and_after_light. Is, however, ' + correction)
        raise ValueError
    return corrected_currents


def plot_sweep(sweep, show_plot=False, plot_interval=None, correction=None, save_fig=False):
    if plot_interval is None:
        plot_interval = auto_interval_to_plot(sweep)
    else:
        assert (type(plot_interval) == list and len(plot_interval) == 2)
    sweep_time = sweep.times
    sweep_voltage = sweep.voltages
    if correction is None:
        sweep_current = sweep.currents
    elif correction == 'pre_light_only' or 'pre_and_after_light':
        sweep_current = correct_currents(sweep, correction)
    else:
        logging.error('corrected should be None / pre_light_only / pre_and_after_light. Is, however, ' + correction)
        raise ValueError
    fig, axs = plt.subplots(2)
    axs[0].plot(sweep_time[plot_interval[0]:plot_interval[1]], sweep_current[plot_interval[0]:plot_interval[1]])
    axs[0].set(xlabel=sweep.times_title, ylabel=sweep.currents_title)
    axs[1].plot(sweep_time[plot_interval[0]:plot_interval[1]], sweep_voltage[plot_interval[0]:plot_interval[1]])
    axs[1].set(xlabel=sweep.times_title, ylabel=sweep.voltages_title)

    for ax in axs.flat:
        ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.grid(alpha=.2)
        ax.axvspan(sweep.t_shutter_on, sweep.t_shutter_off, color='orange', alpha=.3, lw=0)

    if show_plot:
        plt.show()

    if save_fig:
        sweep_path = Path(sweep.which_abf_file())
        analysis_results_folder = sweep.make_output_folder()
        fig.savefig(
            str(analysis_results_folder) + '/' + str(sweep_path.stem) + '_sweep_' + str(sweep.sweep_nr) + '_plot.pdf')
        plt.close()


def plot_all_sweeps(active_abf, show_plot=False, plot_interval=None, correction=None, save_fig=False):
    if plot_interval is None:
        first_sweep = active_abf.get_sweep(0)
        plot_interval = auto_interval_to_plot(first_sweep)
    else:
        assert (type(plot_interval) == list and len(plot_interval) == 2)
    nr_of_sweeps = active_abf.sweep_count()
    fig, ax = plt.subplots(1)
    for i in range(nr_of_sweeps):
        sweep_number = nr_of_sweeps - 1 - i
        sweep_interation = active_abf.get_sweep(sweep_number)
        time = sweep_interation.times
        if correction is None:
            current = sweep_interation.currents
        elif correction == 'pre_light_only' or 'pre_and_after_light':
            current = correct_currents(sweep_interation, correction)
        else:
            logging.error('corrected should be None / pre_light_only / pre_and_after_light. Is, however, ' + correction)
            raise ValueError
        ax.plot(time[plot_interval[0]:plot_interval[1]], current[plot_interval[0]:plot_interval[1]], alpha=.5,
                label="{} mV".format(sweep_interation.input_voltage))
        ax.legend(loc='upper left', prop={'size': 8})
        if sweep_number == 0:
            ax.set(xlabel=sweep_interation.times_title, ylabel=sweep_interation.currents_title)
            ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
            ax.grid(alpha=.2)
            ax.axvspan(sweep_interation.t_shutter_on, sweep_interation.t_shutter_off, color='orange', alpha=.3, lw=0)

    if show_plot:
        plt.show()

    if save_fig:
        abf_path = Path(active_abf.which_abf_file())
        analysis_results_folder = active_abf.make_output_folder()
        if correction is None:
            fig.savefig(str(analysis_results_folder) + '/' + str(abf_path.stem) + '_all_sweeps_not_corrected_plot.pdf')
        else:
            fig.savefig(str(analysis_results_folder) + '/' + str(
                abf_path.stem) + '_all_sweeps_corrected_' + correction + '_plot.pdf')
        plt.close()
