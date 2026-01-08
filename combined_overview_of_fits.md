---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: hnu
    language: python
    name: hnu
---

```python
import matplotlib.pyplot as plt
from hierarchical_nu.fit import StanFit
import numpy as np
import h5py
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import StepPatch
import os
from hnu_paper import DATA_DIR
from pathlib import Path
import astropy.units as u
from scipy.stats.mstats import gmean as geometricmean
from hnu_paper import PAGEWIDTH, TEXTHEIGHT, find_latest_file
import seaborn as sns
import arviz as av
import pandas as pd
from roque_cmap import roque_chill
CMAP = roque_chill()
# module("load", "texlive/2021")
plt.style.use("paper.mplstyle")
PAGEWIDTH = 6.31616
MARGINWIDTH = 1.87831
TEXTWIDTH = 4.2134 
```

```python
colors = CMAP(np.linspace(0, 1, 7))
```

```python
fits = []
am3_fits = []
energy_unit = u.erg
x_energy_unit = u.Hz
area_unit = u.cm**2
E_power = 2.

order = []

meta = pd.read_csv("meta.txt", delimiter="\t|\s{2,}", comment="#", header=0)
meta = np.array(meta)   # shorten this for testing
source_names = meta[:, 0]
alert_ids = meta[:, 3]
alt_names = meta[:, 1]
```

```python
sources = [
    "4FGL J1528.4+2004",
    "4FGL J0509.4+0542",
    "4FGL J0854.0+2753",
    "4FGL J1808.8+3522",
    "4FGL J0158.8+0101",
    "4FGL J0244.7+1316",
    "4FGL J1808.2+3500",
]
```

```python
rows = len(sources)
cols = 2
```

```python
rows
```

```python
E = np.geomspace(1e2, 1e9, 1_000) << u.GeV
plot_E = E.to_value(u.GeV, equivalencies=u.spectral())
x_energy_unit = u.GeV
```

```python
fig = plt.figure(figsize=(PAGEWIDTH, PAGEWIDTH * 4 / 5))
gs = fig.add_gridspec(
    ncols=3,   #   #-parameters + 1 for colourbar
    nrows=2,   # cases to be compared against one another
    width_ratios=[1., 1., 1.,],
    hspace=0.05,
    wspace=0.05)

axs = [fig.add_subplot(gs[i, 0]) for i in range(2)]
#axs.append(fig.add_subplot(gs[2, 0], sharex=axs[0]))

frac = 100
scale = 1.1

helper_lw = 0.8


#### luminosity
xlim = 42, 50
for c, _fits in enumerate(["config.yml", "config_am3_prior.yml"]):
    # ax.set_ylabel(row_labels[c])
    #ax.set_xlim(41.5, 46.5)
    
    i = 0
    n = len(sources) - 1

    for s in reversed(sources):
        
        fit = StanFit.from_file(*find_latest_file(s.lower().replace(" ", "_") + f"/{_fits}"))

        #x, pdf = fit._get_kde("L", transform=lambda x: np.log10((x * u.GeV / u.s).to_value(u.erg / u.s)))
        #max_p = pdf.max()
        #axs[c].plot(x, pdf / max_p * scale + i, color="white", zorder=(len(sources) - i) / len(sources), lw=1)\
        bins = np.linspace(41, 50, (50-41)*8+1, endpoint=True)
        vals, _ = np.histogram(np.log10((fit["L"] * u.GeV / u.s).to_value(u.erg / u.s)), bins=bins, density=True)
        max_p = vals.max()
        axs[c].stairs(vals / max_p * scale + i, bins, zorder=(len(sources) - i) / len(sources), lw=1.5, fill=False, edgecolor="white")
        patch = StepPatch(vals / max_p * scale + i, bins, baseline=i, fill=True, facecolor=colors[n], edgecolor="none", zorder=(len(sources) - i) / len(sources))
        axs[c].add_patch(patch)
        # axs[c].stairs(vals / max_p * scale + i, bins, zorder=(len(sources) - i) / len(sources), lw=1, fill=False, edgecolor="white")
        axs[c].hlines(i, *xlim, color="black", lw=helper_lw, zorder=(len(sources) + .5 - i) / len(sources))
        # axs[c].fill_between(x, i, pdf / max_p * scale + i, facecolor=colors[n], edgecolor="none", zorder=(len(sources) - i) / len(sources))
        # bins = np.concatenate([np.atleast_1d(bins[)])
        # axs[c].fill_between()

        try:
            prior = fit.priors.luminosity[0]
        except:
            prior = fit.priors.luminosity

        x = np.linspace(*xlim, 1_000)
        pdf = prior.pdf_logspace(np.power(10, x) * u.erg / u.s)

        axs[c].plot(
            x,
            i + pdf / max_p * scale,
            alpha=0.5,
            color="grey",
            zorder=(len(sources) - i) / len(sources),
        )
        
        i += 1
        n -= 1
    axs[c].set_xlim(*xlim)
    axs[c].set_ylim(0, i - 1 + scale)
    axs[c].set_yticks(np.arange(.5, i), reversed(sources), rotation=45,)
    axs[c].tick_params(axis="y", right=False)
    if c == 0:
        axs[c].set_xticklabels([])

axs[c].set_xlabel(r"$\log_{10}\left(\frac{L}{\si{\erg\per\second}}\right )$")
    
for ax in axs:
    ax.set_ylim(0, len(sources) + 0.2)
#### E0_src

xlim = 3, 10
axs = [fig.add_subplot(gs[i, 1]) for i in range(2)]

for c, _fits in enumerate(["config.yml", "config_am3_prior.yml"]):
    # ax.set_ylabel(row_labels[c])
    #ax.set_xlim(41.5, 46.5)
    
    i = 0
    n = len(sources) - 1

    for s in reversed(sources):
        
        fit = StanFit.from_file(*find_latest_file(s.lower().replace(" ", "_") + f"/{_fits}"))

        bins = np.linspace(2, 10, (10-2)*8+1, endpoint=True)
        vals, _ = np.histogram(np.log10(fit["E0_src"]), bins=bins, density=True)
        max_p = vals.max()
        
        # axs[c].plot(x, pdf / max_p * scale + i, color="white", zorder=(len(sources) - i) / len(sources), lw=1)
        # axs[c].hlines(i, *xlim, color="black", lw=helper_lw, zorder=(len(sources) + 0.5 - i) / len(sources))
        # axs[c].fill_between(x, i, pdf / max_p * scale + i, facecolor=colors[n], edgecolor="none", zorder=(len(sources) - i) / len(sources))
        axs[c].stairs(vals / max_p * scale + i, bins, zorder=(len(sources) - i) / len(sources), lw=1.5, fill=False, edgecolor="white")
        patch = StepPatch(vals / max_p * scale + i, bins, baseline=i, fill=True, facecolor=colors[n], edgecolor="none", zorder=(len(sources) - i) / len(sources))
        axs[c].add_patch(patch)
        # axs[c].stairs(vals / max_p * scale + i, bins, zorder=(len(sources) - i) / len(sources), lw=1, fill=False, edgecolor="white")
        axs[c].hlines(i, *xlim, color="black", lw=helper_lw, zorder=(len(sources) + .5 - i) / len(sources))

        try:
            prior = fit.priors.E0_src[0]
        except:
            prior = fit.priors.E0_src

        x = np.linspace(*xlim, 1_000)
        pdf = prior.pdf_logspace(np.power(10, x) * u.GeV)

        axs[c].plot(
            x,
            i + pdf / max_p * scale,
            alpha=0.5,
            color="grey",
            zorder=(len(sources) - i) / len(sources),
        )
        
        i += 1
        n -= 1
    axs[c].set_xlim(xlim[0], 9.5)
    axs[c].set_ylim(0, i - 1 + scale)
    axs[c].set_yticks([])
    if c == 0:
        axs[c].set_xticklabels([])
    
axs[c].set_xlabel(r"$\log_{10}\left(\frac{E^\mathrm{src}_\mathrm{peak}}{\si{\GeV}}\right )$")
for ax in axs:
    ax.set_ylim(0, len(sources) + 0.2)

#### Nex_Src
xlim = 0, 10
axs = [fig.add_subplot(gs[i, 2]) for i in range(2)]
for c, _fits in enumerate(["config.yml", "config_am3_prior.yml"]):
    # ax.set_ylabel(row_labels[c])
    #ax.set_xlim(41.5, 46.5)
    
    i = 0
    n = len(sources) - 1

    for s in reversed(sources):
        
        bins = np.arange(0, 20.1, 0.5)
        
        fit = StanFit.from_file(*find_latest_file(s.lower().replace(" ", "_") + f"/{_fits}"))
        vals, _ = np.histogram(fit["Nex_src"], bins=bins, density=True)
        max_p = vals.max()
        axs[c].stairs(vals / max_p * scale + i, bins, zorder=(len(sources) - i) / len(sources), lw=1.5, fill=False, edgecolor="white")
        patch = StepPatch(vals / max_p * scale + i, bins, baseline=i, fill=True, facecolor=colors[n], edgecolor="none", zorder=(len(sources) - i) / len(sources))
        axs[c].add_patch(patch)
        # axs[c].plot(x, pdf / max_p * scale + i, color="white", zorder=(len(sources)-i)/len(sources), lw=1)
        axs[c].hlines(i, *xlim, color="black", lw=helper_lw, zorder=(len(sources) + 0.5 - i) / len(sources))
        # axs[c].fill_between(x, i, pdf / max_p * scale + i, facecolor=colors[n], edgecolor="none", zorder=(len(sources) - i) / len(sources))

        
        i += 1
        n -= 1
    axs[c].set_xlim(*xlim)
    axs[c].set_ylim(0, i - 1 + scale)
    axs[c].set_yticks([])
    if c == 0:
        axs[c].set_xticklabels([])
    
axs[c].set_xlabel(r"$\bar{n}$")

for ax in axs:
    ax.set_ylim(0, len(sources) + 0.2)
    #ax.set_xlim(-0.02, 10)

# fig.savefig("bl_lac_summary.png", bbox_inches="tight", dpi=300)
```

```python

```
