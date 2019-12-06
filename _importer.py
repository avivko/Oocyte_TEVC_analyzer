from _abfAnalysis import *
import glob as glob
from pathlib import Path



def import_single_abf(abf_path):
    file_path_as_object = Path(abf_path)
    assert file_path_as_object.is_file(), 'The given path seems to be invalid (not a file); given path : {} '.format(
        file_path_as_object)
    return ActiveAbf(abf_path)


def import_abfs_from_dic(folder_path, file_name_pattern='*.abf'):
    folder_path_as_object = Path(folder_path)
    assert folder_path_as_object.is_dir(), 'The given path seems to be invalid (not a directory); given path : {} ' \
        .format(folder_path_as_object)
    list_of_abfs = glob.glob(folder_path + file_name_pattern)
    assert list_of_abfs, 'No files were found in the path {} '.format(folder_path)
    list_of_active_abf_objects = [ActiveAbf(i) for i in list_of_abfs]
    return list_of_active_abf_objects


