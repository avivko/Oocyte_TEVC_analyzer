from _abfAnalysis import *
import pandas as pd
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


# Import *_sweeps.csv file
def import_sweeps_from_csv(path):
    file_path_as_object = Path(path)
    assert file_path_as_object.match('*_sweeps.csv'), 'The given path is not a sweeps csv file; given path : {} '. \
        format(file_path_as_object)

    def DF_nparray_to_1_dim_list(nparray):
        return list(nparray.reshape((nparray.size,)))

    df = pd.read_csv(path)
    imported_data = {"voltagesAsNpyArray": df[[col for col in df if "4_voltage" in col]].to_numpy(),
                     "voltagesStdAsNpyArray": df[[col for col in df if "5_SD_of_voltage" in col]].to_numpy(),
                     "currentsAsNpyArray": df[[col for col in df if "2_currents" in col]].to_numpy(),
                     "currentsStdAsNpyArray": df[[col for col in df if "3_SD_of_currents" in col]].to_numpy(),
                     "name": file_path_as_object.stem}
    imported_data["voltages"] = DF_nparray_to_1_dim_list(imported_data["voltagesAsNpyArray"])
    imported_data["currents"] = DF_nparray_to_1_dim_list(imported_data["currentsAsNpyArray"])
    imported_data["voltages_std"] = DF_nparray_to_1_dim_list(imported_data["voltagesStdAsNpyArray"])
    imported_data["currents_std"] = DF_nparray_to_1_dim_list(imported_data["currentsStdAsNpyArray"])

    imported_data["weightsAsNpyArray"] = 1 / (imported_data["currentsStdAsNpyArray"])
    imported_data["weights"] = list(imported_data["weightsAsNpyArray"].reshape((imported_data["weightsAsNpyArray"].size,)))

    return imported_data
