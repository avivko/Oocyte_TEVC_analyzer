from _abfAnalysis import *
from _fitting import *
from _importer import *
import sys
import os
from _loggerInitializer import *


def no_args_dialog():
    print("Welcome to TECV Analyzer")
    print("Please make sure the following modules have been installed and that you are running on python >=3.5:")
    needed_modules = "pyabf", "matplotlib.pyplot", "numpy", "pandas", "lmfit", "pathlib", "glob"
    print(needed_modules)
    py_ver = sys.version_info[0] + sys.version_info[1] / 10
    if py_ver < 3.5:
        raise Exception("Python ver. " + str(py_ver) + " is too old. Python 3.5 or above is required")
    print("checking if installed...")
    for module in needed_modules:
        if not module in sys.modules:
            raise ModuleNotFoundError(module)
    print("modules are indeed installed!")
    print("Please run \'TECV_analyzer.py --options\' for the different options of running this script ")


def options_dialog():
    print('To run the script, please use the command \'--run\' as follows:')
    print('TECV_analyzer.py --run <<run options>> <<abf path>>')
    print(' ')
    print('With the following options:')
    print(' ')
    print('<<run options>> could be ONE of the following:')
    print('\'\' (i.e. none) : will output the corrected all-sweep plots (.pdf) and the current and voltage data files '
          '(.csv)')
    print('\'-u\' (i.e. uncorrected) : will also output the uncorrected all-sweep plots plots (.pdf)')
    print('\'-p\' (i.e. per sweep) : will also output the individual per sweep plots (.pdf)')
    print('\'-v\' (i.e. verbose correction) : will also output the pre-light corrected all-sweep plots (.pdf)')
    print('\'-a\' (i.e. all) : will output all of the above')
    print(' ')
    print('<<run options>> could be ONE of the following:')
    print('\'\' (i.e. none) : the current path from which this script was run will be used')
    print('<path_to_abf_file> : the path to a specific .abf file')
    print('<path_to_folder> : the path to a specific the folder where the to be analyzed .abf files are')
    print(' ')
    print("FYI: The plots and the analyzed currents and voltage data will be placed in an output folder in the given "
          "abf path along with a log file")


def make_log(abf):
    output_folder_path = abf.make_output_folder()
    initialize_logger(str(output_folder_path))


def run(input_option, input_path):
    assert input_option is None or input_option == 'u' or input_option == 'p' or input_option == 'v' \
           or input_option == 'a'
    print('Input: option, path = ' + str(input_option) + ', ' + str(input_path))
    if input_path is None:
        input_path = os.getcwd()
    if Path(input_path).is_file():
        abf_to_analyze = input_path
        imported_single_abf = import_single_abf(abf_to_analyze)
        abfs_as_list = [imported_single_abf]
    elif Path(input_path).is_dir():
        folder_to_analyze = input_path
        imported_abfs = import_abfs_from_dic(folder_to_analyze)
        abfs_as_list = imported_abfs
    else:
        raise ValueError('Bad path:' + str(input_path) + 'could not be found / is incorrect')
    make_log(abfs_as_list[0])
    for abf in abfs_as_list:
        msg = "analyzing file " + abf.which_abf_file() + " ..."
        logging.info(msg)
        if input_option == 'p' or input_option == 'a':
            for i in range(abf.sweep_count()):
                sweep_i = abf.get_sweep(i)
                plot_sweep(sweep_i, save_fig=True)
        if input_option == 'u' or input_option == 'a':
            plot_all_sweeps(abf, save_fig=True)
        try:
            if input_option == 'v' or input_option == 'a':
                plot_all_sweeps(abf, correction='pre_light_only', save_fig=True)
            plot_all_sweeps(abf, correction='pre_and_after_light', save_fig=True)
            abf.export_analyzed_abf_data_to_csv()
        except AssertionError:
            logging.warning('Could not correct the currents in this file. Plotting uncorrected currents and skipping.')
            plot_all_sweeps(abf, save_fig=True)


def main():
    arguments = sys.argv
    nr_of_args = len(arguments) - 1
    if nr_of_args == 0:
        no_args_dialog()
    elif arguments[1] == "--options":
        options_dialog()
    elif arguments[1] == "--run":
        if nr_of_args == 1:
            given_option = None
            given_path = None
        elif nr_of_args == 2:
            if arguments[2][0] == '-':
                given_option = arguments[2][1]
                given_path = None
            elif arguments[2][0] != '-':
                given_option = None
                if arguments[2][-1] == '/':
                    given_path = arguments[2]
                else:
                    given_path = arguments[2] + '/'
            else:
                raise ValueError('given arguments are not available. please see --options')
        elif (nr_of_args == 3 and arguments[2][0]) == '-':
            given_option = arguments[2][1]
            if arguments[3][-1] == '/':
                given_path = arguments[3]
            else:
                given_path = arguments[3] + '/'
        else:
            raise ValueError('given arguments are not available. please see --options')
        run(given_option, given_path)
    else:
        raise ValueError('given arguments are not available. please see --options')


if __name__ == '__main__':
    main()
