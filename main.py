from AbfAnalysis import *
import matplotlib.pyplot as plt
import pyabf as pyabf
import numpy as np

abfToAnalyze = '/home/kormanav/Dokumente/TEVC_24_10_2019/Messungen_Aviv/2019_10_23_0053.abf'

abf = ActiveAbf(abfToAnalyze)
plot_all_sweeps(abf, [1900,4500])