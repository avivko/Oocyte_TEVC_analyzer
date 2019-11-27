from AbfAnalysis import *
from fitting import *
import matplotlib.pyplot as plt
import pyabf as pyabf
import numpy as np



### fucntions ###


#abfToAnalyze = '/home/kormanav/Dokumente/TEVC_29_10_2019/2019_10_29_0004.abf'
# abfToAnalyze = '/home/kormanav/Dokumente/TEVC_13_11_2019/2019_11_13_0085.abf'
# abf = ActiveAbf(abfToAnalyze)
# plot_all_sweeps(abf, [400,6500])

abfToAnalyze = '/home/kormanav/Dokumente/Messungen_Aviv/2019_11_26_0003.abf'

#abfToAnalyze = '/Volumes/Transcend/TEVC_15_11_2019/2019_11_15_0021.abf'

abf = ActiveAbf(abfToAnalyze)
sweepnr = 6
sweep1 = sweep(abfToAnalyze, sweepnr)
plotInterval = [2000, 16600]
plot_sweep(sweep1, plot_interval=plotInterval)
plot_all_sweeps(abf, plot_interval=plotInterval)
plot_all_sweeps(abf, plot_interval=plotInterval, corrected='pre_light_only')
plot_all_sweeps(abf, plot_interval=plotInterval, corrected='pre_and_after_light')
print(get_voltage_changes(abf))
#plot_sweep(sweep1, corrected='pre_and_after_light')
#plot_all_sweeps(abf,[2000, 16000])
#plot_all_sweeps(abf,[1700, 16000],corrected='pre_and_after_light')
