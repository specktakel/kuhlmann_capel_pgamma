```python
from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.utils.cache import Cache
import argparse
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import ligo.skymap.plot
from pathlib import Path
import colorsys
import matplotlib.colors as mc
from roque_cmap import roque_chill
import h5py
from roque_cmap import roque_chill
import seaborn as sns
from scipy import stats
from contextlib import contextmanager
from hnu_paper import DATA_DIR, PAGEWIDTH, COLUMNWIDTH, BEAMERWIDTH
import arviz as av
PAGEWIDTH = 6.31616
BEAMERWIDTH = 5.2
MARGINWIDTH = 1.87831
TEXTWIDTH = 4.2134 
plt.style.use("../beamer.mplstyle")   
module("load", "texlive/2021")
```

```python
cmap = roque_chill().reversed()
```


```python

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("--files", nargs="*", default=[])
parser.add_argument("--skyllh", action="store_true")
parser.add_argument("--freq", nargs=1)
args = parser.parse_args()
files = args.files


```

```python
latest = not files
#assumes dirname / config_file_name

dir_name = Path(os.path.split(config)[0])
config_file = Path(os.path.split((config))[1])
config_name = os.path.splitext(config_file)[0]
skyllh_file = f"config_skyllh.h5"
skyllh = args.skyllh
```

```python
freq = Path(args.freq)
```

```python
@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
```

```python
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
```

```python

from hierarchical_nu.fit import StanFit
from hierarchical_nu.utils.roi import ROIList
from hierarchical_nu.source.parameter import Parameter

ROIList.clear_registry()
Parameter.clear_registry()

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
```

```python
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
CL = np.array([0.68, 0.95])#[::-1]

hdi = {key: [find_hdi(transformations[c](fit._fit_output[key].T), _).flatten() for _ in CL] for c, key in enumerate(keys)}
print(hdi)
kdes = {key: av.kde(transformations[c](fit._fit_output[key])) for c, key in enumerate(keys)}

priors = {"src_index": fit.priors.src_index, "L": fit.priors.luminosity}

if skyllh:
    with h5py.File("pl_4fgl_j0509.4+0542/" + skyllh_file, "r") as f:
        gamma_vals = f["gamma_vals"][()]
        ns_vals = f["ns_vals"][()]
        delta_ts = f["delta_ts"][()]


    chi2_68_quantile = stats.chi2.ppf(0.68, df=2)
    #chi2_90_quantile = stats.chi2.ppf(0.90, df=2)
    chi2_95_quantile = stats.chi2.ppf(0.95, df=2)
    levels_skyllh = [chi2_68_quantile, chi2_95_quantile]


skyllh_color = "black"
levels_hnu = 1 - np.array([0.0, 0.68, 0.95])[::-1]

levels = 1 - np.array([0.0, 0.68, 0.95])[::-1]


colors = plt.cm.viridis(np.linspace(0, 1, 10))
# hnu_colors = [colors[0], colors[4], colors[6]]
l = [1, 0.4]
hnu_colors = []
for i in l:
    color = colorsys.rgb_to_hls(*mc.to_rgb("C0"))
    color = colorsys.hls_to_rgb(color[0], 1 - i * (1 - color[1]), color[2])
    hnu_colors.append(color)


# Start with a square Figure.
fig = plt.figure(figsize=(BEAMERWIDTH * 0.8, 3))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal Axes and the main Axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.0, hspace=0.0)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_x.yaxis.set_tick_params(labelleft=False)
ax_x.set_yticks([])
ax_y = fig.add_subplot(gs[1, 1], sharey=ax)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_y.set_xticks([])
# Draw the scatter plot and marginals.


src_index_chains = fit._fit_output["src_index"].flatten()
Nex_ps_chains = fit._fit_output["Nex_src"].flatten()



sns.kdeplot(
    x=src_index_chains,
    y=Nex_ps_chains,
    levels=levels,
    ax=ax,
    #cmap=cmap,
    fill=True,
    color=hnu_colors[0],
    alpha=.8,
)
mapped_colours = ax.collections[0]._mapped_colors[::-1]

if skyllh:
    ax.contour(
        gamma_vals,
        ns_vals,
        delta_ts,
        levels_skyllh,
        colors=skyllh_color,
        linestyles=["--", "-.", ":"]
    )
if freq:
    errs = ref["src_index"]
    x = errs[0]
    xerr = np.abs(errs[0] - errs[1:3])[:, np.newaxis]
    errs = ref["Nex_src"]
    y = errs[0]
    yerr = np.abs(errs[0] - errs[1:3])[:, np.newaxis]
    ax.scatter(limits[0, 0], limits[0, 1], marker="+", color="black", zorder=20)


for c, (key, a, o) in enumerate(zip(keys, [ax_x, ax_y], ["h", "v"])):
    #ax = axs[c]
    kde = kdes[key]
    plotx = kde[0]
    ploty = kde[1]
    if o == "v":
        plotx, ploty = ploty, plotx
        ylim = (ploty.min(), ploty.max())
    else:
        xlim = (plotx.min(), plotx.max())
    a.plot(plotx, ploty, color="black")
    if o == "v":
        xmax = a.get_xlim()[1]
        a.set_xlim(0, xmax)
    else:
        ymax = a.get_ylim()[1]
        a.set_ylim(0, ymax)
    if key == "L":
        a.plot(kde[0], priors[key].pdf(kde[0]*u.erg / u.s), color="grey")
    elif key == "src_index":
        a.plot(kde[0], priors[key].pdf(kde[0]), color="grey")
    for c_, HDI in enumerate(hdi[key]):
        print(key, HDI)
        support = np.linspace(HDI[0], HDI[1], 1_000)
        if o == "h":
            a.fill_between(support, np.zeros_like(support), np.interp(support, *kde), color=mapped_colours[c_], edgecolor="none", zorder=1-c_)
        else:
            a.fill_betweenx(support, np.zeros_like(support), np.interp(support, *kde), color=mapped_colours[c_], edgecolor="none", zorder=1-c_)

    if freq:
        try:
            errs = ref[key]
            xerr = np.abs(errs[1:3])[:, np.newaxis]

            plotx = errs[0]
            if o == "v":
                kwarg = dict(yerr=xerr)
                ploty = np.sum(a.get_xlim())*0.45
                plotx, ploty = ploty, plotx
            else:
                ploty = np.sum(a.get_ylim())*0.45
                kwarg = dict(xerr=xerr)

            err = a.errorbar(plotx, ploty, color="black", capsize=4, fmt="+", label=r"MLE $\pm 1\sigma$",**kwarg)
        except KeyError:
            pass

    #if o == "v":
    #    a.legend()
    #mle_handle = a.get_legend().legend_handles[0]
    #a.get_legend().remove()
    #ax.set_yticks([])
    #ax.set_xlabel(label[c])
print(xlim, ylim)
ax.set_xlim(1, 3.2)
ax.set_ylim(0, 52)    

ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\bar{n}$")

handles = [
    patches.Patch(facecolor="none", edgecolor="none"),
    Line2D([0], [0], color="grey"),
    #patches.Patch(facecolor=mapped_colours[2], edgecolor="none"),
    patches.Patch(facecolor=mapped_colours[1], edgecolor="none"),
    patches.Patch(facecolor=mapped_colours[0], edgecolor="none"),
    patches.Patch(facecolor="none", edgecolor="none"),
    err,
    #Line2D([0], [0], color="black", ls=":"),
    Line2D([0], [0], color="black", ls="-."),
    Line2D([0], [0], color="black", ls="--"),
]
labels = [r"\texttt{hnu}", r"prior", r"95\% credibility", r"68\% credibility", r"\texttt{SkyLLH}", r"MLE $(\pm 1 \sigma)$", r"95\% CL", r"68\% CL",]
fig.legend(
    handles=handles, labels=labels,
    ncols=2, loc="outside lower center",
    bbox_to_anchor=(0.5, 0.91),
    handlelength=3,
)
```

```python

```
