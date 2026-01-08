from pathlib import Path
import os
from contextlib import contextmanager
import numpy as np

from .utils import FileConfig


### This file defines some utility functions

file_config = FileConfig.load()


# Directories where fits are saved
DATA_DIR = Path(file_config.data_dir)
MC_DIR = Path(file_config.mc_dir)
ANG_SYS_DIR = Path(file_config.ang_sys_dir)


# Lengths used in latex
PAGEWIDTH = 6.00117
COLUMNWIDTH = 2.80728
TEXTHEIGHT = 9.2018
BEAMERWIDTH = 5.5129


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        
# locate the latest fit file of some config in the data directory    
def find_latest_file(config_name, latest: bool = True):
    dir_name = config_name.split("/")[-2]
    all_files = os.listdir(DATA_DIR / dir_name)
    config = config_name.split("/")[-1].rstrip(".yml")
    # print(dir_name)
    # print(config)
    fit_files = [f for f in all_files if ".h5" in f]
    matching = []
    # I refuse to learn regex for this
    # or ask chatgpt for some regex
    timestamps = []
    for f in fit_files:
        base = os.path.splitext(f)[0]
        split = base.split(str(config))
        if len(split) == 1:
            continue
        if not "fit_" == split[0]:
            continue
        if len(split) == 2:
            if split[1] == "":
                matching.append(f)
                timestamps.append(0)
                continue
            try:
                timestamps.append(int(split[1].lstrip("_")))
                matching.append(f)  
            except ValueError:
                continue
    if latest:
        timestamps = np.array(timestamps)
        load = [DATA_DIR / dir_name / matching[np.argmax(timestamps)]]

    else:
        load = [DATA_DIR / dir_name / f for f in matching]
        
    return load



# translate break energy to peak energy, cf. Eq. (3) of the paper
def peak_energy(break_energy):
    return np.exp(1 / 0.7) * break_energy


# translate back
def break_energy(peak_energy):
    return peak_energy / np.exp(1 / 0.7)
