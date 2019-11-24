from AbfAnalysis import *
from fitting import *
import matplotlib.pyplot as plt
import pyabf as pyabf
import numpy as np
import pyabf.tools.memtest


### fucntions ###


#abfToAnalyze = '/home/kormanav/Dokumente/TEVC_29_10_2019/2019_10_29_0004.abf'
# abfToAnalyze = '/home/kormanav/Dokumente/TEVC_13_11_2019/2019_11_13_0085.abf'
# abf = ActiveAbf(abfToAnalyze)
# plot_all_sweeps(abf, [400,6500])

#abfToAnalyze = '/home/kormanav/Dokumente/TEVC_15_11_2019/2019_11_15_0020.abf'

abfToAnalyze = '/Volumes/Transcend/TEVC_15_11_2019/2019_11_15_0021.abf'

abf = ActiveAbf(abfToAnalyze)
sweepnr = 7
sweep1 = sweep(abfToAnalyze, sweepnr)
plotInterval = [2000,15000]
plot_sweep(sweep1,plot_interval=plotInterval)
plot_sweep(sweep1,plot_interval=plotInterval,corrected='pre_light_only')
plot_sweep(sweep1,plot_interval=plotInterval,corrected='pre_and_after_light')
plot_all_sweeps(abf,plot_interval=plotInterval,corrected='pre_light_only')
plot_all_sweeps(abf,plot_interval=plotInterval,corrected='pre_and_after_light')
#plot_sweep(sweep1, corrected='pre_and_after_light')
#plot_all_sweeps(abf,[2000, 16000])
#plot_all_sweeps(abf,[1700, 16000],corrected='pre_and_after_light')
