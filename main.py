from AbfAnalysis import *
from fitting import *
from importexport import *
import matplotlib.pyplot as plt
import os
import pyabf as pyabf
import numpy as np



### fucntions ###

#abfToAnalyze = '/Volumes/PENDISK/bsp_messungen/2019_10_29_0003.abf'
#folderToAnalyze = '/Volumes/PENDISK/bsp_messungen/'
folderToAnalyze = '/home/kormanav/Dokumente/bsp_messungen/'
abfToAnalyze = '/home/kormanav/Dokumente/bsp_messungen/2019_10_29_0066.abf'

imported_abfs = import_abfs_from_dic(folderToAnalyze)
imported_single_abf = import_single_abf(abfToAnalyze)


sweepnr = 6
sweep1 = sweep(imported_single_abf.which_abf_file(), sweepnr)
plot_sweep(sweep1)
plot_all_sweeps(imported_single_abf)
plot_all_sweeps(imported_single_abf, corrected='pre_light_only')
plot_all_sweeps(imported_single_abf, corrected='pre_and_after_light')
# print(get_voltage_changes(imported_single_abf))

