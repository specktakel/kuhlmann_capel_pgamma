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

### Contribution of the sources to the diffuse flux of astro neutrinos

Sum all the fluxes of the sources for each spectral model/prior separately

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import astropy.units as u
# module("load", "texlive/2021")
from hierarchical_nu.fit import StanFit
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from hnu_paper import find_latest_file
import pandas as pd
import colorsys
from matplotlib import colors as mc
import arviz as av
from hnu_paper import COLUMNWIDTH, PAGEWIDTH
import matplotlib.patches as mpatches
from scipy import convolve
plt.style.use("../paper.mplstyle")
PAGEWIDTH = 6.31616
MARGINWIDTH = 1.87831
TEXTWIDTH = 4.2134 
```

```python
meta = pd.read_csv("meta.txt", delimiter="\t|\s{2,}", comment="#", header=0, engine="python")
meta = np.array(meta)   # shorten this for testing
source_names = meta[:, 0]
alert_ids = meta[:, 3]
alt_names = meta[:, 1]
```

```python
source_names
```

```python
flux_grid = np.zeros((1_000, 16_000))
```

```python
E = np.geomspace(1e2, 1e9, 1_000) << u.GeV
```

```python jupyter={"outputs_hidden": true}
# This is used to extract the flux values from the fits
restart = False
for j, s in enumerate(source_names):
    if j == 0 or j == 1:
        continue
    print(s)

    for c in ["/config.yml", "/config_am3_prior.yml"]:
        fit = StanFit.from_file(
            *find_latest_file(s.lower().replace(" ", "_") + c)
        )
        fit.diagnose()
        fit._calculate_flux_grid()
        np.savez_compressed(
            "/viper/ptmp1/jdk/hnu_paper_fits/hnu_paper_fits/" + s.lower().replace(" ", "_") + c.rstrip(".yml") + "_flux.npz",
            flux=fit._flux_grid.squeeze().to_value(1 / u.GeV / u.s / u.m**2)
        )
        del fit

    fit = StanFit.from_file(
        *find_latest_file("pl_" + s.lower().replace(" ", "_") + "/config.yml")
    )
    fit.diagnose()
    fit._calculate_flux_grid()
    np.savez_compressed(
        "/viper/ptmp1/jdk/hnu_paper_fits/hnu_paper_fits/" + "pl_" + s.lower().replace(" ", "_") + "/config_flux.npz",
        flux=fit._flux_grid.squeeze().to_value(1 / u.GeV / u.s / u.m**2)
    )
    del fit
    


```

```python
E = np.geomspace(1e2, 1e9, 1_000) << u.GeV
```

```python
rng = np.random.default_rng()
```

```python
### build added flux distribution

flux_pl = np.zeros((1_000, 16_000)) << u.GeV / u.s / u.m**2 / u.sr
for c, s in enumerate(source_names):
    print(s)
    d = "pl_" + s.lower().replace(" ", "_")
    temp = rng.choice((np.load("/viper/ptmp1/jdk/hnu_paper_fits/hnu_paper_fits/" + d + "/config_flux.npz")["flux"] / u.GeV / u.s / u.m**2 * E[:, np.newaxis]**2 / 2 / np.pi / u.sr).to_value(u.GeV / u.s / u.m**2 / u.sr), axis=1, size=16_000)
    flux_pl += (temp * u.GeV / u.s / u.m**2 / u.sr)
```

```python
np.save("pl_bootstrapped_flux.npy", flux_pl.to_value(u.GeV / u.s / u.m**2 / u.sr))
```

```python
flux_pgamma = np.zeros((1_000, 16_000)) << u.GeV / u.s / u.m**2 / u.sr

for s in source_names:
    print(s)
    d = s.lower().replace(" ", "_")
    temp = rng.choice((np.load("/viper/ptmp1/jdk/hnu_paper_fits/hnu_paper_fits/" + d + "/config_flux.npz")["flux"] / u.GeV / u.s / u.m**2 * E[:, np.newaxis]**2 / 2 / np.pi / u.sr).to_value(u.GeV / u.s / u.m**2 / u.sr), axis=1, size=16_000)
    flux_pgamma += (temp * u.GeV / u.s / u.m**2 / u.sr)
```

```python
np.save("pgamma_bootstrapped_flux.npy", flux_pgamma.to_value(u.GeV / u.s / u.m**2 / u.sr))
```

```python
flux_pgamma_inf = np.zeros((1_000, 16_000)) << u.GeV / u.s / u.m**2 / u.sr

for s in source_names:
    if s == "4FGL J0854.0+2753" or s == "4FGL J1808.8+3522":
        continue
    print(s)
    d = s.lower().replace(" ", "_")
    temp = rng.choice((np.load("/viper/ptmp1/jdk/hnu_paper_fits/hnu_paper_fits/" + d + "/config_am3_prior_flux.npz")["flux"] / u.GeV / u.s / u.m**2 * E[:, np.newaxis]**2 / 2 / np.pi / u.sr).to_value(u.GeV / u.s / u.m**2 / u.sr), axis=1, size=16_000)
    flux_pgamma_inf += (temp * u.GeV / u.s / u.m**2 / u.sr)
```

```python
# np.save("pgamma_inf_bootstrapped_flux.npy", flux_pgamma_inf.to_value(u.GeV / u.s / u.m**2 / u.sr))
```

```python
# Icecube measurement https://arxiv.org/pdf/2402.18026
IC_diffuse = {
    "Abbasi et al.~(2024), best fit and $1\sigma$ CL":
        {
            "phi0": 1.68,
            "C0":  1e-18 / u.GeV / u.s / u.cm**2 / u.sr,
            "gamma": 2.58,
            "E0": 100 * u.TeV,
            "phi0_e": 0.22,
            "gamma_e": 0.10,
            "Esens": [3 * u.TeV, 5.5 * u.PeV],
        },
    "Abbasi et al.~(2022), best fit and $1\sigma$ CL":
        {
            "phi0": 1.44,
            "C0":  1e-18 / u.GeV / u.s / u.cm**2 / u.sr,
            "gamma": 2.37,
            "E0": 100 * u.TeV,
            "phi0_e": 0.26,
            "gamma_e": 0.09,
            "Esens": [15 * u.TeV, 5 * u.PeV],
        },
    }
            

```

```python
# powerlaw
def pl(E, gamma, phi, E0):
    return np.power(E / E0, -gamma) * phi

def pl_deriv(E, gamma, phi, E0, gamma_err, phi_err):
    return np.sqrt(
        (phi_err * np.power(E / E0, -gamma))**2 + (pl(E, gamma, phi, E0) * np.log((E / E0).to_value(1)) * gamma_err)**2
    )
```

```python
unit = u.GeV
```

```python
# coverage of the sky by our source selection
# RA -> 2pi
# dec -> half the sky, realistically because we have to incorporate some finite size ROI, so source has to be above 0degrees declination
```

```python
# Load saved flux grids
flux_pl = np.load("pl_bootstrapped_flux.npy") * u.GeV / u.s / u.m**2 / u.sr
flux_pgamma = np.load("pgamma_bootstrapped_flux.npy") * u.GeV / u.s / u.m**2 / u.sr
flux_pgamma_inf = np.load("pgamma_inf_bootstrapped_flux.npy") * u.GeV / u.s / u.m**2 / u.sr
```

```python
flux_pl == flux_pgamma
```

```python
ic_high_energy_limit = np.power(10, np.loadtxt("icecube_high_energy_flux_limit.txt", delimiter=","))
```

```python
km3 = np.power(10, np.loadtxt("km3net_single_event.txt", delimiter=","))
```

```python
km3_energy = km3[:2, 0]
km3_single_data_point = km3[2:, 1]
km3_central_energy = km3[-1, 0]
```

```python
#for i in range(3):
#    print(km3_single_data_point[4 + i] - km3_single_data_point[3], km3_single_data_point[2 - i] - km3_single_data_point[3])
#print(km3_single_data_point)
```

```python
km3_energy, km3_central_energy
```

```python
km3_single_data_point
```

```python
km3_single_data_point[3], km3_single_data_point
```

```python
quants_pl = np.quantile(flux_pl, [0.5 - 0.6827/2, 0.5+0.6827/2], axis=1).to_value(u.GeV / u.cm**2 / u.s / u.sr)
quants_pgamma = np.quantile(flux_pgamma, [0.5 - 0.6827/2, 0.5+0.6827/2], axis=1).to_value(u.GeV / u.cm**2 / u.s / u.sr)
quants_pgamma_inf = np.quantile(flux_pgamma_inf, [0.5 - 0.6827/2, 0.5+0.6827/2], axis=1).to_value(u.GeV / u.cm**2 / u.s / u.sr)


pl_ul = np.quantile(flux_pl, 0.9, axis=1).to_value(u.GeV / u.cm**2 / u.s / u.sr)
pgamma_ul = np.quantile(flux_pgamma, 0.9, axis=1).to_value(u.GeV / u.cm**2 / u.s / u.sr)
pgamma_inf_ul = np.quantile(flux_pgamma_inf, 0.9, axis=1).to_value(u.GeV / u.cm**2 / u.s / u.sr)
```

```python
# define an object that will be used by the legend
class MulticolorPatch():
    def __init__(self, colors):
        self.colors = colors
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler():
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c, 
                           edgecolor='none'))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch

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
#fig, ax = fits[0].plot_flux_band(2., credible_interval=[0.68, 0.95], energy_unit=unit, figsize=(7, 5))
#fig, ax = fits[1].plot_flux_band(2., credible_interval=[0.68, 0.95], energy_unit=unit, ax=ax, color="C1")

fig, ax = plt.subplots(figsize=(PAGEWIDTH, 3/5 * PAGEWIDTH))
ax.set_xscale("log")
ax.set_yscale("log")

# plot bowtie
handles = []
labels = []

#rectangle = Rectangle((0, 1), 0.1, 0.1, facecolor="grey", alpha=0.3, edgecolor="none")
# line = Line2D([0], [0], color="grey")

#handles.append(line)
#labels.append(r"$90\%$ UL")

#handles.append(rectangle)
#labels.append(r"$68\%$ CL/CR")

one_sigma = (
       mpatches.Patch(facecolor="C0", edgecolor="none", alpha=0.3),
       mpatches.Patch(facecolor="C1", alpha=0.3, edgecolor="none")
)


for i in range(3):
    #break
    l = [0.5, 0.7, 0.9][::-1]
    color = lighten_color("black", l[i])
    scat = ax.errorbar(
        km3_central_energy,
        km3_single_data_point[3],
        xerr=np.abs(km3_central_energy - km3_energy)[:, np.newaxis],
        yerr=np.abs(np.array([[km3_single_data_point[4+i] - km3_single_data_point[3]], [km3_single_data_point[3] - km3_single_data_point[2-i]]])),
        #yerr=np.array([[5e-8], [1e-7]]),
        color=color,
        alpha=0.5,
        # label=r"KM3$-$230213, KM3NeT Collaboration (2025)" if i == 2 else None,
        fmt="none",
        zorder=2,
        
    )
    if i == 0:
        handles.append(scat)
        labels.append(r"KM3NeT Collaboration~(2025), $1\sigma, 2\sigma, 3\sigma$ CL")

# handles.append(one_sigma)
# labels.append(r"$1\sigma$ flux uncertainty")

for c, (label, diff) in enumerate(IC_diffuse.items()):
    
    flux = pl(E, diff["gamma"], diff["phi0"] * diff["C0"], diff["E0"]).to_value(1 / unit / u.s / u.cm**2 / u.sr)
    err = pl_deriv(E, diff["gamma"], diff["phi0"] * diff["C0"], diff["E0"], diff["gamma_e"],  diff["phi0_e"] * diff["C0"]).to_value(1 / unit / u.s / u.cm**2 / u.sr)
    
    ax.fill_between(
        E.to_value(u.GeV)[(E < diff["Esens"][1]) & (E > diff["Esens"][0])],
        (E.to_value(unit)**2 * (flux-err))[(E < diff["Esens"][1]) & (E > diff["Esens"][0])],
        (E.to_value(unit)**2 * (flux+err))[(E < diff["Esens"][1]) & (E > diff["Esens"][0])], alpha=0.3,
                   color=f"C{c}",
                   # color="none",
                   edgecolor="none",
    )
    l = ax.plot(E.to_value(u.GeV)[(E < diff["Esens"][1]) & (E > diff["Esens"][0])], (E.to_value(unit)**2*flux)[(E < diff["Esens"][1]) & (E > diff["Esens"][0])],
           color=f"C{c}", ls="--")
    labels.append(label)
    box = mpatches.Patch(facecolor=f"C{c}", alpha=0.3, edgecolor="none")
    handles.append((l[0], box))

    # ax.plot(E.to_value(u.GeV), (E.to_value(unit)**2 * flux), ls="--", color=f"C{c}",)


he_ic_lim = ax.errorbar(ic_high_energy_limit[:, 0], ic_high_energy_limit[:, 1] / 3, color="black", ls="--", uplims=True, yerr=ic_high_energy_limit[:, 1] / 3 * 0.5)
labels.append("Abbasi et al.~(2025), 90\% CL upper limit")
handles.append(he_ic_lim[0])
handles.append(mpatches.Patch(color="none"))
labels.append("")
labels.append("")
handles.append(mpatches.Patch(color="none"))
#fig, ax = fits[0].plot_flux_band(2., credible_interval=[0.68, 0.95], energy_unit=unit, ax=ax)
#fig, ax = fits[1].plot_flux_band(2., credible_interval=[0.68, 0.95], energy_unit=unit, ax=ax, color="C1")
#ax.plot(E_sens.to_value(u.GeV), (E_sens**2 * pl(E_sens, gamma, phi0*C0, E0)).to_value(u.TeV / u.s / u.m**2 / u.sr) * np.pi * 4, color="C0")


#for c, f in enumerate(fits):
#    label = "individual sources" if c == 0 else ""
#    f.plot_flux_band(2.0, credible_interval=0.9, energy_unit=unit, area_unit=u.m**2, upper_limit=True, ax=ax, color="C1", marker="None", label=label)



#pl_handle = ax.plot(E.to_value(u.GeV), np.quantile(flux_pl.to_value(u.GeV / u.cm**2 / u.s / u.sr), 0.90, axis=1), color="C2")
#pg_handle = ax.plot(E.to_value(u.GeV), np.quantile(flux_pgamma.to_value(u.GeV / u.cm**2 / u.s / u.sr), 0.90, axis=1), color="C2")
#pg_inf_handle = ax.plot(E.to_value(u.GeV), np.quantile(flux_pgamma_inf.to_value(u.GeV / u.cm**2 / u.s / u.sr), 0.90, axis=1), color="C3")
# ax.plot(E.to_value(u.GeV), np.quantile(flux_pgamma_inf_source_removed, 0.90, axis=1), label=r"90\% UL $p\gamma$ Fit \#2 w/o J0854", color="C5")
# ax.fill_between(E.to_value(u.GeV), flux_pl_68[:, 0], flux_pl_68[:, 1], color="C2", alpha=0.2, edgecolor="none")

# labels.append("power law")
# handles.append(pl_handle[0])



#labels.append(r"Fit \#2")
#handles.append(pg_inf_handle[0])
#labels.append(r"Fit \#1")
#handles.append(pg_handle[0])
# ax.fill_between(E.to_value(u.GeV), flux_pgamma_68[:, 0], flux_pgamma_68[:, 1], color="C2", alpha=0.2, edgecolor="none")


ax.fill_between(E.to_value(u.GeV), quants_pl[0], quants_pl[1], color="C2", alpha=0.2, edgecolor="none", label="Power law, 68\% credibility")
ax.fill_between(E.to_value(u.GeV), quants_pgamma[0], quants_pgamma[1], color="C3", alpha=0.2, edgecolor="none", label=r"$p\gamma$, weakly informative, 68\% credibility")
ax.fill_between(E.to_value(u.GeV), quants_pgamma_inf[0], quants_pgamma_inf[1], color="C4", alpha=0.2, edgecolor="none", label=r"$p\gamma$, informative, 68\% credibility")
# ax.plot(E.to_value(u.GeV), quants_pl.to_value(u.GeV / u.s / u.cm**2 / u.sr)[0], color="C2")
# ax.plot(E.to_value(u.GeV), quants_pl.to_value(u.GeV / u.s / u.cm**2 / u.sr)[1], color="C2")
#hand, lab = ax.get_legend_handles_labels()
#handles += reversed(hand)
#labels += reversed(lab)


# ax.fill_between(E.to_value(u.GeV), flux_pgamma_68[:, 0], flux_pgamma_68[:, 1], color="C3", alpha=0.2, edgecolor="none")
# ax.plot(E.to_value(u.GeV), quants_pgamma.to_value(u.GeV / u.s / u.cm**2 / u.sr)[0], color="C2")
# ax.plot(E.to_value(u.GeV), quants_pgamma.to_value(u.GeV / u.s / u.cm**2 / u.sr)[1], color="C2")


# ax.fill_between(E.to_value(u.GeV), flux_pgamma_inf_68[:, 0], flux_pgamma_inf_68[:, 1], color="C3", alpha=0.2, edgecolor="none")
# ax.fill_between(E.to_value(u.GeV), quants_pgamma_inf.to_value(u.GeV / u.s / u.cm**2 / u.sr)[0], color="C3")


arrowprops = dict(width=1, headwidth=4, headlength=4)
ax.annotate("IceCube starting tracks (2024)", (1e4, 7e-8), (2e4,2e-7), arrowprops=arrowprops | dict(color="C0"), ha="center", va="bottom", fontsize=10)
ax.annotate("IceCube 9.5yr tracks (2022)", (2e4, 2e-8), (1.5e4,4e-9), arrowprops=arrowprops | dict(color="C1"), ha="center", va="top", fontsize=10)
ax.annotate("IceCube upper limit (2025)", (1.6e8, 2e-9), (8e6, 5e-11), arrowprops=arrowprops | dict(color="black"), ha="left", va="top", fontsize=10)
ax.annotate("Power law", (3e4, 3e-10), (2e4, 3e-11), arrowprops=arrowprops | dict(color="C2"), ha="center", va="top", fontsize=10)
ax.annotate(r"$p\gamma$, informative", (2e7, 2e-8), (1e6, 5e-8), arrowprops=arrowprops | dict(color="C4"), ha="left", va="top", fontsize=10)
ax.annotate(r"$p\gamma$, weakly informative", (3e5, 8e-10), (3e5, 9e-11), arrowprops=arrowprops | dict(color="C3"), ha="center", va="top", fontsize=10)
ax.annotate("KM3NeT (2025)", (2e8, 7e-8), (3e7, 3e-7), arrowprops=arrowprops | dict(color="grey"), ha="center", va="top", fontsize=10)


ax.set_xlim(1e3, 1e9)
ax.set_ylim(1e-11, 1e-6)

ax.set_xlabel(r"$E~[\si{\GeV}]$")
ax.set_ylabel(r"$\nu_\mu$ energy flux~[\si{\giga\eV\per\centi\meter\squared\per\second\per\steradian}]")
ax.grid(alpha=0.3)

"""
fig.legend(
    [
        mpatches.Patch(edgecolor="none", facecolor="none"),
        mpatches.Patch(edgecolor="none", facecolor="C0"),
        mpatches.Patch(edgecolor="none", facecolor="C1"),
        mpatches.Patch(edgecolor="none", facecolor="none"),
        mpatches.Patch(edgecolor="none", facecolor="none"),
         *bands,
        mpatches.Patch(edgecolor="none", facecolor="none"),
        mpatches.Patch(edgecolor="none", facecolor="none"),
        dashed,
        solid,
        scatter,
    ],
    [
        "Fit",
        r"Fit \#1",
        r"Fit \#2",
        "",
        "Energy flux",
        r"95\% credibility",
        r"68\% credibility",
        "",
        r"$E_\mathrm{peak}^\mathrm{obs}$---$\max{\left(E^2 \dv{N}{E}\right)}$",
        r"95\% credibility",
        r"68\% credibility",
        r"R+ prediction",
    ],
    loc="outside upper center",
    #frameon=True,
    #framealpha=0.8,
    edgecolor="none",
    ncols=3,
    bbox_to_anchor=(.5, 0.98),
    
)
"""

```

```python
fig.savefig("combined_flux.png", dpi=300, bbox_inches="tight")
```

```python

```
