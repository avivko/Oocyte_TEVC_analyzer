from AbfAnalysis import *
from fitting import *
from importexport import *




### fucntions ###

abfToAnalyze = '/Volumes/PENDISK/bsp_messungen/2019_10_29_0003.abf'
folderToAnalyze = '/Volumes/PENDISK/bsp_messungen/'
#folderToAnalyze = '/home/kormanav/Dokumente/bsp_messungen/'
#abfToAnalyze = '/home/kormanav/Dokumente/bsp_messungen/2019_10_29_0066.abf'

imported_abfs = import_abfs_from_dic(folderToAnalyze)
imported_single_abf = import_single_abf(abfToAnalyze)

for i in imported_abfs:
    sweepnr = 6
#   sweep1 = sweep(i.which_abf_file(), sweepnr)
#   plot_sweep(sweep1,save_fig=True)
    plot_all_sweeps(i,save_fig=True)
    plot_all_sweeps(i, correction='pre_light_only',save_fig=True)
    plot_all_sweeps(i, correction='pre_and_after_light',save_fig=True)
    # print(get_voltage_changes(imported_single_abf))

