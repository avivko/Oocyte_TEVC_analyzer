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
    imported_data = {
        "voltages": df[[col for col in df if "4_voltage" in col]].to_numpy(),
        "voltagesSD": df[[col for col in df if "5_SD_of_voltage" in col]].to_numpy(),
        "currents": df[[col for col in df if "2_currents" in col]].to_numpy(),
        "currentsSD": df[[col for col in df if "3_SD_of_currents" in col]].to_numpy()
    }
    imported_data["voltagesAsList"] = DF_nparray_to_1_dim_list(imported_data["voltages"])
    imported_data["currentsAsList"] = DF_nparray_to_1_dim_list(imported_data["currents"])
    imported_data["voltagesSDAsList"] = DF_nparray_to_1_dim_list(imported_data["voltagesSD"])
    imported_data["currentsSDAsList"] = DF_nparray_to_1_dim_list(imported_data["currentsSD"])

    imported_data["SDweights"] = 1 / (imported_data["voltagesSD"] * imported_data["currentsSD"])
    imported_data["SDweightsAsList"] = list(imported_data["SDweights"].reshape((imported_data["SDweights"].size,)))

    return imported_data
