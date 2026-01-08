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
from hierarchical_nu.fit import StanFit
from hnu_paper import DATA_DIR
from hierarchical_nu.utils.config import HierarchicalNuConfig
from hierarchical_nu.utils.config_parser import ConfigParser
from pathlib import Path
from omegaconf import OmegaConf
from hierarchical_nu.events import Events
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
import astropy.units as u
import seaborn as sns
import arviz as av
from matplotlib.lines import Line2D
from matplotlib import patches
from hnu_paper import COLUMNWIDTH, PAGEWIDTH, find_latest_file
module("load", "texlive/2021")
import colorsys
from matplotlib import colors as mc
PAGEWIDTH = 6.31616
MARGINWIDTH = 1.87831
TEXTWIDTH = 4.2134 
```

```python
plt.style.use("paper.mplstyle")
```

```python
dir_name = "4fgl_j0509.4+0542"
config_name = "config"
am3_inf_load = find_latest_file(dir_name + "/config_am3_prior.yml") 
load = find_latest_file("pl_4fgl_j0509.4+0542/config.yml")
am3_load = find_latest_file(dir_name + "/config.yml")

pl_load = find_latest_file("pl_4fgl_j0509.4+0542/config.yml")
```

```python
load, am3_load, am3_inf_load
```

```python
pl = StanFit.from_file(*pl_load)
am3_fit = StanFit.from_file(*am3_load)
am3_inf = StanFit.from_file(*am3_inf_load)
```

```python
p = np.array(pl._get_event_classifications())[:, 0]

```

```python
p.argsort()[::-1]
```

```python
pl.events.energies.argsort()[::-1][:10]
```

```python
pl.events.mjd[19]
```

```python
plt.vlines(pl.events.mjd.mjd, 0, p)
```

```python
gmean(pl["E"][..., 3358].flatten()), np.exp(np.average(np.log10(pl["E"][..., 3358].flatten())))
```

```python
fits = [pl, am3_fit, am3_inf]
```

```python
for f in fits:
    print("Nex av", np.average(f["Nex_src"]))
    print("P(N>1)", np.sum(f["Nex_src"] >= 1) / f["Nex_src"].size)
```

```python
# Find number of events above certain association thresholds
thresholds = np.array([5, 20, 50]) * 0.01   # in percent
N = []
for fit in [pl, am3_inf]:
    prob = np.array(fit._get_event_classifications())[:, 0]
    N.append([np.sum(prob > p) for p in thresholds])
```

```python
N
```

```python
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
```

```python
upper = np.loadtxt("4fgl_j0509.4+0542/ic_ul.dat", delimiter=",")
lower = np.loadtxt("4fgl_j0509.4+0542/ic_ll.dat", delimiter=",")
flux = np.loadtxt("4fgl_j0509.4+0542/mm_nu_flux.dat", delimiter=",")

best_fit_skyllh = np.loadtxt("../logparabola_txs/rodrigues_spectrum_nu_F_nu.csv", delimiter=',')
freq = np.power(10, best_fit_skyllh[:, 0]) << u.Hz
best_fit_E = freq.to(u.GeV, equivalencies=u.spectral())
best_fit_flux = (np.power(10, best_fit_skyllh[:, 1]) << u.erg / u.cm**2 / u.s)


el = (np.power(10, lower[:, 0]) * u.eV).to(u.GeV)
eu = (np.power(10, upper[:, 0]) * u.eV).to(u.GeV)

ef = (np.power(10, flux[:, 0]) * u.eV).to(u.GeV)

fl = np.power(10, lower[:, 1]) * u.erg / u.cm**2 / u.s
fu = np.power(10, upper[:, 1]) * u.erg / u.cm**2 / u.s

f = np.power(10, flux[:, 1]) * u.erg / u.cm**2 / u.s
```

```python
fig, axs = plt.subplots(1, 2, figsize=(PAGEWIDTH, 2.), gridspec_kw=dict(hspace=0, wspace=0.05))


for c, ax in enumerate(axs):
    if c == 1:
        label = r"R+, \texttt{SkyLLH} 68\% CL"
        ax.plot(el.to_value(u.GeV), fl.to_value(u.erg / u.cm**2 / u.s), color="black", label=label, ls="--")
        
    else:
        ax.plot(el.to_value(u.GeV), fl.to_value(u.erg / u.cm**2 / u.s), color="black", ls="--")
    ax.plot(eu.to_value(u.GeV), fu.to_value(u.erg / u.cm**2 / u.s), color="black", ls="--")
    ax.plot(best_fit_E.to_value(u.GeV), best_fit_flux.to_value(u.erg / u.cm**2 / u.s), color="grey", label=r"R+, \texttt{SkyLLH} best fit")
    if c == 1:
        label = r"\texttt{am3}"
        ax.plot(ef.to_value(u.GeV), f.to_value(u.erg / u.cm**2 / u.s), color="red", label=label)
    else:
        ax.plot(ef.to_value(u.GeV), f.to_value(u.erg / u.cm**2 / u.s), color="red")
        
legend_patches = [patches.Patch(edgecolor="none", facecolor="none")]

for c, (ax, fi) in enumerate(zip(axs, [am3_fit, am3_inf])):
    CR = reversed([0.5, 0.683, 0.95,])
    lighten = [0.2, 0.4, 0.6]
    for cr, l in zip(CR, lighten):
        color = lighten_color("C0", l)
        fi.plot_flux_band(
            2.,
            cr,
            energy_unit=u.erg,
            area_unit=u.cm**2,
            figsize=(COLUMNWIDTH, 2),
            ax=ax,
            color=color,
            alpha=1.,
        )
        if c == 0:
            legend_patches.append(patches.Patch(edgecolor="none", facecolor=color))
    fi.plot_peak_energy_flux(ax, energy_unit=u.erg)

"""
for cont, c, ci in zip(contours, colors, confidence):
    try:
        ax.plot(np.power(10, cont[:, 0]), np.power(10, cont[:, 1]), color=c, label=f"{ci*100:n}\% CI")
    except:
        continue
"""
for ax in axs:
    ax.set_xlim(2e3, 1e9)
    ax.set_ylim(1e-14, 1e-9);
    ax.set_xlabel(r"$E~[\si{\GeV}]$")
    
axs[0].set_ylabel(r"$E^2\frac{\dd[3]{n}}{\dd{E}\dd{A}\dd{t}}~[\si{\erg\per\centi\meter\squared\per\second}]$")
axs[1].set_ylabel("")
axs[1].set_yticklabels([])
legend = axs[0].get_legend()
lines = legend.legend_handles
axs[0].get_legend().remove()
handles = axs[1].get_legend().legend_handles
handles.insert(0, Line2D([0], [0], color="none"))
handles.insert(4, Line2D([0], [0], color="none"))
axs[1].get_legend().remove()
labels = [
    "Rodrigues et al.~(2024)",
    r"\texttt{SkyLLH} 68\% CL",
    r"\texttt{SkyLLH} best fit",
    r"flux prediction",
    r"$E_\mathrm{peak}^\mathrm{obs}$---$\max{\left(E^2 \frac{\dd[3]{n}}{\dd{E}\dd{A}\dd{t}}\right)}$",
    r"95\% credibility",
    r"68\% credibility",
    r"50\% credibility",
]
handles += legend_patches
labels += [
    "energy flux",
    "95\% credibility",
    "68\% credibility",
    "50\% credibility",
]

labels = labels[4:] + labels[0:4]
handles = handles[4:] + handles[0:4]
fig.legend(handles, labels, loc="outside left center", ncols=1, bbox_to_anchor=(0.9, 0.5))

for ax in axs:
    xlim = ax.get_xlim()
    xticks = ax.get_xticks()
    ticklabels = ax.get_xticklabels()
    ticklabels[-1] = ""
    ticklabels[-2] = ""
    ax.set_xticks(xticks, ticklabels)
    ax.set_xlim(*xlim)

#axs[0].legend(legend.legend_handles, legend._label, bbox_to_anchor=(0.5, 0.5))
fig.savefig("dissertation_plots/4fgl_j0509.4+0542_peak_flux_comparison_thesis.png", bbox_inches="tight", dpi=300)
```

```python
### Comparison between PL, pgamma uninformative and pgamma informative

# overplot 1) IC170922A energy posteriors, Nex trace

idx = np.argwhere(np.isclose(pl.events.mjd.mjd, 58018.871186)).flatten()[0]

fig, axs = plt.subplots(
    2,
    1,
    figsize=(TEXTWIDTH, 3),
    gridspec_kw=dict(hspace=0.5)
)
# axs = [axs]

labels = ["power law", "$p\gamma$ uninformative", "$p\gamma$ informative"]

handles = [Line2D([0], [0], color=f"C{_}") for _ in range(3)]
#handles = [Line2D([0], [0], color=f"C{_}") for _ in [0, 2]]
# Nex = [np.average(fit["Nex_src"]), am3_fit, np.average(am3_inf["Nex_src"])]

for c, (fit, label) in enumerate(zip([pl, am3_fit, am3_inf], labels)):
    #supp, kde = fit._get_kde("Nex_src")
    #axs[0].plot(supp, kde, label=label)
    #if fit is am3_fit:
    #    continue
    axs[0].hist(fit["Nex_src"].flatten(), bins=np.arange(0, np.ceil(fit["Nex_src"].flatten().max()) + 1, 1.), density=True, histtype="step",color=f"C{c}")
ylim = axs[0].get_ylim()

for c, (fit, label) in enumerate(zip([pl, am3_fit, am3_inf], labels)):
    #if fit is am3_fit:
    #    continue
    mean = np.average(fit["Nex_src"].flatten())
    axs[0].vlines(mean, *ylim, color=f"C{c}", ls=":")
    print(c)
    #axs[0].text(mean+0.05, 0.2, rf"$\langle\bar{{n}}\rangle = {mean:.1f}$", fontsize=8, rotation=-30, rotation_mode="anchor", ha="left", va="bottom")
# labels = ["power law","$p\gamma$"]
axs[0].set_xlim(0, 40)
axs[0].legend(handles, labels)
axs[0].set_yticks([]);
axs[0].set_xlabel(r"$\bar{n}$");
# fig.savefig("4fgl_j0509.4+0542/txs_model_comparison_pl_pgamma_inf.png", dpi=300, bbox_inches="tight")
#fig.savefig("4fgl_j0509.4+0542/icrc_posterior_comparison.png", dpi=300, bbox_inches="tight")




for c, (fit, label) in enumerate(zip([pl, am3_fit, am3_inf], labels)):
    #supp, kde = fit._get_kde("E", idx, transform=lambda x: np.log10(x))
    #axs[1].plot(np.power(10, supp), kde,)
    #if fit is am3_fit:
    #    continue
    print(c)
    n, bins = np.histogram(np.log10(fit["E"][..., idx].flatten()), bins=np.linspace(3, 9, 61))
    axs[1].stairs(n, np.power(10, bins), color=f"C{c}")
ylim = axs[1].get_ylim()
for c, (fit, label) in enumerate(zip([pl, am3_fit, am3_inf], labels)):
    #if fit is am3_fit:
    #    continue
    mean = np.power(10, np.average(np.log10(fit["E"][..., idx].flatten())))
    print(mean)
    axs[1].vlines(mean, *ylim, color=f"C{c}", ls=":")

axs[1].set_xlabel(r"$E~[\si{\GeV}]$")
axs[1].set_xscale("log")
axs[1].set_yticks([]);
axs[1].set_ylim(bottom=0)
axs[0].set_ylim(bottom=0)
axs[1].set_xlim(5e3, 1e9)

fig.savefig("dissertation_plots/4fgl_j0509.4+0542_txs_model_comparison.png", dpi=300, bbox_inches="tight")
# fig.savefig("4fgl_j0509.4+0542/txs_model_comparison_pl_pgamma_inf_prop.png", dpi=300, bbox_inches="tight")

```

```python
probs = np.array(pl._get_event_classifications())[:, 0]
alert = probs.argmax()
highlight = np.zeros(pl.events.N)
highlight[alert] = 1.
```

```python
fig, ax = plt.subplots(figsize=(3.5, 3))
ax, mapper = pl._plot_energy_posterior(ax, pl.get_src_position(), 0, 5*u.deg, "lin", highlight=highlight)
fig.colorbar(mapper, ax=ax, label="association probability")
```

```python
fig, axs = pl.plot_energy_and_roi(highlight=highlight, figsize=(TEXTWIDTH, TEXTWIDTH * 2 / 5))
axs[1].set_xlabel(r"$E~[\si{\GeV}]$")
axs[1].set_ylim(0,)
axs[1].set_xlim(1e2, 1e9)
# fig.colorbar(mapper, ax=ax, label="association probability")
```

```python
fig, ax = plt.subplots(figsize=(3.5, 3))
ax, mapper = am3_inf._plot_energy_posterior(ax, am3_inf.get_src_position(), 0, 5*u.deg, "lin", highlight=highlight)
fig.colorbar(mapper, ax=ax, label="association probability")
```

```python

```
