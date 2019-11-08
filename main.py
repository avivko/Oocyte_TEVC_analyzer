from AbfAnalysis import *
from fitting import *
import matplotlib.pyplot as plt
import pyabf as pyabf
import numpy as np
import pyabf.tools.memtest

#abfToAnalyze = '/home/kormanav/Dokumente/TEVC_29_10_2019/2019_10_29_0004.abf'
abfToAnalyze = '/home/kormanav/Dokumente/TEVC_24_10_2019/2019_10_23_0053.abf'
abf = ActiveAbf(abfToAnalyze)
plot_all_sweeps(abf, [500,4500])


