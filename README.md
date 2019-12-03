# Oocyte_TEVC_analyzer

Oocyte_TVEC_analyzer is a program made to automatically analyze data from two-electrode voltage clamp recordings (.abf binary files) collected with the program [Axon™pCLAMP™ Clampex](http://mdc.custhelp.com/app/answers/detail/a_id/18779/~/axon%E2%84%A2pclamp%E2%84%A2-10-electrophysiology-data-acquisition-%26-analysis-software-download)
meant to be run over the terminal. It is meant to help automate your workflow and help make even mediocre data useful without any extra effort.

Right now it could do the following:
* Nicely plot the individual current sweeps in a double plot that also shows the voltage as a function of time
* Nicely plot all the sweeps together 
* Best-fit-based baseline-correction for the photocurrents (either based on the currents before the light is turned on or both before the light as well as after it) and plotting
* Exporting of all of the analyzed data, including real clamp voltages and steady-state currents, into .csv files which could be used for further analysis and fancy plotting 

## Installation

1. Make sure to use python 3.5 or a newer (3.6 or newer is recommended) to run this script. Get it here if you don't have it already: https://www.python.org/downloads/ .

2. Clone the repository to your machine 

3. Make sure the following modules are installed for the environment you are running this script on: [pyabf](https://pypi.org/project/pyabf/), [matplotlib](https://pypi.org/project/matplotlib/), [numpy](https://pypi.org/project/numpy/), [pandas](https://pypi.org/project/pandas/), [lmfit](https://pypi.org/project/lmfit/), [pathlib](https://pypi.org/project/pathlib/), [glob](https://pypi.org/project/glob2/) . 
You can install them using [pip](https://pip.pypa.io/en/stable/) like this:

```bash
pip install numpy
```

You should now be all set!

## Usage

You can access the program by running the following command from your local repository:
```bash
python TEVC_analyzer.py
```
Or from anywhere (for example from a folder where you have saved you .abf files) by specifing the path to your repository:
```bash
python PATH/TO_REOPOSITORY/TEVC_analyzer.py
```
Which will prompt a dialog that will make sure you have set up everything correctly and guide you through different possible options when running this program.



Updates and more exemplary data will follow, so follow this page and dont forget to fetch new versions every once in a while!
Enjoy!
