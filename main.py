from AbfAnalysis import *
from fitting import *
from importexport import *
import matplotlib.pyplot as plt
import pyabf as pyabf
import numpy as np



### fucntions ###

abfToAnalyze = '/Volumes/PENDISK/bsp_messungen/2019_10_29_0003.abf'
folderToAnalyze = '/Volumes/PENDISK/bsp_messungen/'

imported_abfs = import_abfs_from_dic(folderToAnalyze)
imported_single_abf = import_single_abf(abfToAnalyze)


sweepnr = 6
sweep1 = sweep(imported_single_abf.which_abf_file(), sweepnr)
print(sweep1.t_shutter_on)
plotInterval = [2000, 16600]
# plot_sweep(sweep1, plot_interval=plotInterval)
# plot_all_sweeps(abf, plot_interval=plotInterval)
# plot_all_sweeps(abf, plot_interval=plotInterval, corrected='pre_light_only')
# plot_all_sweeps(abf, plot_interval=plotInterval, corrected='pre_and_after_light')
# print(get_voltage_changes(abf))
#plot_sweep(sweep1, corrected='pre_and_after_light')
#plot_all_sweeps(abf,[2000, 16000])
#plot_all_sweeps(abf,[1700, 16000],corrected='pre_and_after_light')
