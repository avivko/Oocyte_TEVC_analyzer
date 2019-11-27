from AbfAnalysis import *
import glob as glob


def import_abfs_from_dic(folder_path, file_name_pattern='*'):
    list_of_abfs = glob.glob(folder_path + file_name_pattern)
    list_of_active_abf_objects = [ActiveAbf(i) for i in list_of_abfs]
    return list_of_active_abf_objects
