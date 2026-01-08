from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.utils.cache import Cache
import argparse
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import ligo.skymap.plot
from pathlib import Path
from hnu_paper import DATA_DIR, PAGEWIDTH, COLUMNWIDTH, cd

plt.style.use("paper.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("-s", "--setup", action="store_true")
parser.add_argument("-r", "--random", nargs="*", default=[-1], type=int)
parser.add_argument("-e", "--events", action="store_true")
parser.add_argument("-mufac", type=float)
parser.add_argument("-sigma", type=float)
args = parser.parse_args()
setup = args.setup
seed = args.random
ev = args.events
mufac = args.mufac
sigma = args.sigma
#assumes dirname / config_file_name
dir_name = Path(os.path.split(args.config)[0])
config_file = Path(os.path.split((args.config))[1])
config_name = os.path.splitext(config_file)[0]
print(setup)
print(seed)
print(dir_name)

with cd(dir_name):
    Cache.set_cache_dir("../.cache")
    from hierarchical_nu.utils.config import HierarchicalNuConfig
    from hierarchical_nu.utils.config_parser import ConfigParser
    from hierarchical_nu.source.parameter import Parameter
    from hierarchical_nu.utils.roi import ROIList
    from hierarchical_nu.priors import LuminosityPrior, LogNormalPrior

    config = HierarchicalNuConfig.from_path(config_file)
    # config.prior_config.L.mu *= mufac
    # config.prior_config.L.sigma = sigma
    parser = ConfigParser(config)
    sources = parser.sources
    parser.ROI
    dm = parser.detector_model
    obs_time = parser.obs_time
    events = parser.events
    print(events.N)
    if ev:
        sys.exit()
    """
    sim = parser.create_simulation(sources, dm, obs_time) 
    sim.precomputation()
    sim._get_expected_Nnu(sim._get_sim_inputs())
    lumi = Parameter.get_parameter("luminosity")
    N_ps = sim._Nex_et.sum(axis=0)[0]
    """
    fit = parser.create_fit(sources, events, dm, obs_time)
    # set lumi prior to predict 1 event
    stan_kwargs = dict(parser.stan_kwargs)
    # stan_kwargs["iterations"] = 2000
    if seed != [-1]:
        stan_kwargs["seed"] = seed
    
    if setup:
        fit.generate_stan_code()
        fit.compile_stan_code()
        sys.exit()
    fit.precomputation()
    inits = dict()
    
    N_ps = len(sources.point_source)
    
    if config.parameter_config.share_src_index:
        inits["src_index"] = 2.3
        inits["beta_index"] = 0.05
        inits["E0_src"] = 1e6
    else:
        inits["src_index"] = [2.3] * N_ps
        inits["beta_index"] = [0.05] * N_ps
        inits["E0_src"] = [1e6] * N_ps
        
    if config.parameter_config.share_L:
        inits["L"] = 1e48
    else:
        inits["L"] = [1e48] * N_ps
        
    inits["diff_flux"] = 2.26e-13
    inits["F_atmo"] = 0.3
    inits["diff_index"] = 2.5
    inits["E"] = [5e4] * events.N
    # fit.generate_stan_code()
    # fit.compile_stan_code()
    try:
        fit.setup_stan_fit()
    except:
        fit.generate_stan_code()
        fit.compile_stan_code()
    fit.run(inits=inits, show_progress=False, show_console=True, save_warmup=False, parallel_chains=4, **stan_kwargs)
    """
    except:
        # if something fails try to re-generate stan code
        fit.generate_stan_code()
        fit.compile_stan_code()
        fit.run(inits=inits, **stan_kwargs)
    """
    os.makedirs(DATA_DIR / dir_name, exist_ok=True)
    # path = fit.save(DATA_DIR / dir_name / Path(f"rerun_fit_{config_name}_mu_{mufac}_sigma_{sigma}.h5"), save_warmup=False)
    path = fit.save(DATA_DIR / dir_name / Path(f"fit_{config_name}.h5"), save_warmup=False)
    fit.diagnose()
    print(str(path))
    SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]
    # SLURM_ARRAY_TASK_ID = os.environ["SLURM_ARRAY_TASK_ID"]
    filename = f"rerun_{config_name}_datafiles_{SLURM_JOB_ID}.txt"
    with open(filename, "a") as f:
        f.write(str(path)+"\n")
    # fit.save_csvfiles(f"csv_{SLURM_JOB_ID}")
    """
    # Plot results in the repo, not at DATA_DIR
    N_ps = len(fit.sources.point_source)
    if N_ps > 1:
        for i in range(N_ps):
            fig, axs = fit.plot_energy_and_roi(center=i, figsize=(PAGEWIDTH, 2.5))
            fig.savefig(f"{config_name}_energy_and_roi_{i}.pdf")
    else:
        fig, axs = fit.plot_energy_and_roi(figsize=(PAGEWIDTH, 2.5))
        fig.savefig(f"{config_name}_energy_and_roi.pdf")
    
    fig, axs = fit.plot_trace_and_priors(fit._def_var_names+["Nex_src"])
    fig.tight_layout()
    fig.savefig(f"{config_name}_trace_and_priors.pdf", bbox_inches="tight")
    """
