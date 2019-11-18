import matplotlib.pyplot as plt
import pyabf as pyabf
import numpy as np
from lmfit import Model

class ActiveAbf:
    def __init__(self, abf_file):
        self._abf_data = pyabf.ABF(abf_file)
        self._abf_file_path = abf_file
        self._data_points_per_sec = self._abf_data.dataRate
        self._sweeps = self._abf_data.sweepCount

    def sweep_count(self):
        return self._abf_data.sweepCount

    def which_abf_file(self):
        return self._abf_file_path

    def get_sweep_voltages(self):
        sweep_voltages = {}
        for sweepNumber in range(self._sweeps):
            self._abf_data.setSweep(sweepNumber)
            sweep_voltages[sweepNumber] = self._abf_data.sweepC[round(len(self._abf_data.sweepC) / 2)]
        return sweep_voltages

    def get_raw_abf_data(self):
        some_data = {}
        for sweepNumber in range(self._sweeps):
            self._abf_data.setSweep(sweepNumber)
            some_data[sweepNumber] = {
                'sweep number': sweepNumber,
                'sweep currents (ADC)': self._abf_data.sweepY,
                'sweep input voltages (DAC)': self._abf_data.sweepC,
                'sweep times (seconds)': self._abf_data.sweepX
            }
        return some_data


class sweep(ActiveAbf):
    def __init__(self, abf_file, sweep_nr, interval=None):
        super().__init__(abf_file)
        self._abf_data.setSweep(sweep_nr)
        self.t_shutter_on = self._abf_data.sweepEpochs.p1s[2] * self._abf_data.dataSecPerPoint
        self.t_shutter_off = self._abf_data.sweepEpochs.p2s[2] * self._abf_data.dataSecPerPoint
        if interval is None:
            interval = [0, -1]
        else:
            assert (type(interval) == list and len(interval) == 2), 'sweep interval should be a list like [t_0,t_last]'
        self._currents = self._abf_data.sweepY[interval[0]:interval[1]]
        self._currents_title = self._abf_data.sweepLabelY
        self._times = self._abf_data.sweepX[interval[0]:interval[1]]
        self._times_title = self._abf_data.sweepLabelX
        self._input_voltage = self._abf_data.sweepC[interval[0]:interval[1]]
        self._input_voltage_title = 'Digital Input Clamp Voltage (mV)'
        self._abf_data.setSweep(sweep_nr, 1)
        self._voltages = self._abf_data.sweepY[interval[0]:interval[1]]
        self._voltages_title = self._abf_data.sweepLabelY
        self._abf_data.setSweep(sweep_nr, 2)
        self._shutter = self._abf_data.sweepY[interval[0]:interval[1]]
        self._shutter_title = 'Shutter Voltage (V)'

    def get_sweep_data(self):
        return {
            'currents': self._currents,
            'currents title': self._currents_title,
            'times': self._times,
            'times title': self._times_title,
            'voltages': self._voltages,
            'voltages title': self._voltages_title,
            'shutter': self._shutter,
            'shutter title': self._shutter_title,
            'shutter on': self.t_shutter_on,
            'shutter off': self.t_shutter_off,
            'input clamp voltage': self._input_voltage,
            'input clamp voltage title': self._input_voltage_title
        }


def plot_sweep(sweep):
    sweep_data = sweep.get_sweep_data()
    time = sweep_data['times']
    current = sweep_data['currents']
    voltage = sweep_data['voltages']
    fig, axs = plt.subplots(2)
    axs[0].plot(time, current)
    axs[0].set(xlabel=sweep_data['times title'], ylabel=sweep_data['currents title'])
    axs[1].plot(time, voltage)
    axs[1].set(xlabel=sweep_data['times title'], ylabel=sweep_data['voltages title'])

    for ax in axs.flat:
        ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.grid(alpha=.2)
        ax.axvspan(sweep_data['shutter on'], sweep_data['shutter off'], color='orange', alpha=.3, lw=0)

    plt.show()


def plot_all_sweeps(ActiveAbf, sweep_interval=None):
    if sweep_interval is None:
        sweep_interval = [0, -1]
    fig, axs = plt.subplots(2)
    nr_of_sweeps = ActiveAbf.sweep_count()
    color_idx = np.linspace(0, 1, nr_of_sweeps)
    for i in range(nr_of_sweeps):
        sweepNumber = nr_of_sweeps -1 - i
        sweep_interation = sweep(ActiveAbf.which_abf_file(), sweepNumber, sweep_interval)
        sweep_data = sweep_interation.get_sweep_data()
        time = sweep_data['times']
        current = sweep_data['currents']
        voltage = sweep_data['voltages']
        print('the type is',time.dtype is 'float64')
        axs[0].plot(time, current, alpha=.5,label="{} mV".format(sweep_data['input clamp voltage'][round(len(sweep_data['input clamp voltage']) / 2)]))
        axs[1].plot(time, voltage, alpha=.5)
        axs[0].legend()
        if sweepNumber == 0:
            axs[0].set(xlabel=sweep_data['times title'], ylabel=sweep_data['currents title'])
            axs[1].set(xlabel=sweep_data['times title'], ylabel=sweep_data['voltages title'])
            for ax in axs.flat:
                ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
                ax.grid(alpha=.2)
                ax.axvspan(sweep_data['shutter on'], sweep_data['shutter off'], color='orange', alpha=.3, lw=0)
    plt.show()


