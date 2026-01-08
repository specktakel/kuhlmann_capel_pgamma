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
from roque_cmap import roque_chill
import h5py
import seaborn as sns
from scipy import stats
from contextlib import contextmanager
from hnu_paper import DATA_DIR, PAGEWIDTH, COLUMNWIDTH, BEAMERWIDTH
plt.style.use("paper.mplstyle")    

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("--files", nargs="*", default=[])
parser.add_argument("--joint-posterior", action="store_true")
parser.add_argument("--skyllh", action="store_true")
parser.add_argument("--freq", nargs=1)
parser.add_argument("--ROI", action="store_true")
parser.add_argument("--kde", action="store_true")
args = parser.parse_args()
files = args.files
try:
    freq = args.freq[0]
except TypeError:
    freq = []

latest = not files
#assumes dirname / config_file_name
dir_name = Path(os.path.split(args.config)[0])
config_file = Path(os.path.split((args.config))[1])
config_name = os.path.splitext(config_file)[0]
skyllh_file = f"config_skyllh.h5"


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        
def find_hdi(samples, CL, resolution=1_000):
    upper_low = 1 - CL
    lower_up = CL
    # print(upper_low, lower_up)
    lower_grid = np.linspace(0, upper_low, resolution)
    upper_grid = np.linspace(lower_up, 1, resolution)
    grid = np.vstack((lower_grid, upper_grid))
    #print(lower_grid.size)
    #print(grid.shape)
    intervals = np.quantile(samples, grid.T)
    #print(intervals)
    width = np.diff(intervals)
    position = np.argmin(width)
    return intervals[position]

with cd(dir_name):
    Cache.set_cache_dir("../.cache")

    from hierarchical_nu.fit import StanFit
    
    if files:
        load = [DATA_DIR / dir_name / f for f in files]
        
    else:
    
        all_files = os.listdir(DATA_DIR / dir_name)
        fit_files = [f for f in all_files if ".h5" in f]

        matching = []
        # I refuse to learn regex for this
        # or ask chatgpt for some regex
        timestamps = []
        for f in fit_files:
            base = os.path.splitext(f)[0]
            split = base.split(str(config_name))
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

    print(load)
    fit = StanFit.from_file(*load)
    fit.diagnose()

    if args.kde:
        fig, axs = fit.plot_trace_and_priors()
        fig.tight_layout()
        fig.savefig(f"{config_name}_trace_and_priors.pdf", bbox_inches="tight")
    
        try:
            fig, axs = fit.plot_trace_and_priors(["L", "E0_src"], transform=True)
            fig.savefig(f"{config_name}_L_E0_trace.pdf")
        except KeyError:
            # Possibly using a power law?
            pass

    N_ps = len(fit.sources.point_source)
    if args.ROI:
        if N_ps > 1:
            for i in range(N_ps):
                fig, axs = fit.plot_energy_and_roi(center=i, figsize=(PAGEWIDTH, 2.5))
                fig.savefig(f"{config_name}_energy_and_roi_{i}.pdf")

        else:
            fig, axs = fit.plot_energy_and_roi(
                figsize=(PAGEWIDTH, 2.5)
            )
            fig.savefig(f"{config_name}_energy_and_roi.pdf", bbox_inches="tight")

        
    if args.joint_posterior:
        # Plot high-level parameters only
        # Only for comparison of power-law fits with skyllh
        import arviz as av
        if freq:
            limits = np.loadtxt(freq)
            ref_nex = limits[:, 1]
            ref_index = limits[:, 0]

            ref = {"src_index": ref_index, "Nex_src": ref_nex}

        keys = [
            # "L",
            "src_index",
            "Nex_src"
        ]
        label = [
            # r"$L~[\si{\erg\per\second}]$",
            r"$\gamma$",
            "Nex"
        ]
        transformations = [
            # lambda x: x * (u.GeV / u.erg).to(1),
            lambda x: x,
            lambda x: x
        ]
        CL = [0.5, 0.683]

        hdi = {key: [find_hdi(transformations[c](fit._fit_output[key].T), _).flatten() for _ in CL] for c, key in enumerate(keys)}
        print(hdi)
        kdes = {key: av.kde(transformations[c](fit._fit_output[key])) for c, key in enumerate(keys)}

        priors = {"src_index": fit.priors.src_index, "L": fit.priors.luminosity}

        fig, axs = plt.subplots(nrows=len(keys), ncols=1, figsize=(COLUMNWIDTH, 2), gridspec_kw={"hspace": .60})

        for c, key in enumerate(keys):
            ax = axs[c]
            kde = kdes[key]
            ax.plot(*kde)
            if key == "L":
                ax.plot(kde[0], priors[key].pdf(kde[0]*u.erg / u.s), color="grey")
            elif key == "src_index":
                ax.plot(kde[0], priors[key].pdf(kde[0]), color="grey")
            for HDI in hdi[key]:
                print(key, HDI)
                support = np.linspace(HDI[0], HDI[1], 1_000)
                ax.fill_between(support, np.zeros_like(support), np.interp(support, *kde), color="C0", alpha=0.3, edgecolor="none")

            if freq:
                try:
                    errs = ref[key]
                    xerr = np.abs(errs[0] - errs[1:3])[:, np.newaxis]
                    ax.errorbar(errs[0], np.sum(ax.get_ylim())*0.45, xerr=xerr, color="black", capsize=4, fmt="x")
                except KeyError:
                    pass
            ax.set_yticks([])
            ax.set_xlabel(label[c])
        fig.savefig(f"{config_name}_high_level_comparison.pdf", bbox_inches="tight")
        

        src_index_chains = fit._fit_output["src_index"]
        Nex_ps_chains = fit._fit_output["Nex_src"]
        if args.skyllh:
            with h5py.File(skyllh_file, "r") as f:
                gamma_vals = f["gamma_vals"][()]
                ns_vals = f["ns_vals"][()]
                delta_ts = f["delta_ts"][()]

            limits = np.loadtxt("freq_limits.dat")

            chi2_68_quantile = stats.chi2.ppf(0.68, df=2)
            chi2_90_quantile = stats.chi2.ppf(0.90, df=2)
            chi2_95_quantile = stats.chi2.ppf(0.95, df=2)
            levels_skyllh = [chi2_68_quantile, chi2_90_quantile, chi2_95_quantile]

        levels_hnu = 1 - np.array([0.0, 0.68, 0.9, 0.95])[::-1]

        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        hnu_colors = [colors[0], colors[4], colors[6]]
        skyllh_color = "grey"
        
        fig, ax = plt.subplots()
        fig.set_size_inches(COLUMNWIDTH, COLUMNWIDTH * 4 / 5)


        levels = 1 - np.array([0.0, 0.68, 0.9, 0.95])[::-1]

        if args.skyllh:
            ax.contour(
                gamma_vals,
                ns_vals,
                delta_ts,
                levels_skyllh,
                colors=skyllh_color,
                alpha=0.7,
            )
            ax.scatter(limits[0, 0], limits[0, 1], marker="+", color="black", zorder=20)

        sns.kdeplot(
            x=src_index_chains[0],
            y=Nex_ps_chains[0],
            levels=levels,
            ax=ax,
            color=hnu_colors[0],
            fill=True,
            #alpha=.8,
        )

        # ax.set_title("NGC 1068")
        ax.set_xlim(1., 3.)
        ax.set_ylim(0., 50)

        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\bar{N}$")


        fig.savefig(f"{config_name}_2d_comparison.pdf", bbox_inches="tight")
