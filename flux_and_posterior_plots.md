### Create plots of energy fluxes and $E_\mathrm{peak}$ vs. $\max(\nu F_\nu(E_\mathrm{peak})$ contours compare against predicted values

```python
import matplotlib.pyplot as plt
from hierarchical_nu.fit import StanFit
import numpy as np
import h5py
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import os
from hnu_paper import DATA_DIR
from matplotlib import ticker
from pathlib import Path
import astropy.units as u
from scipy.stats.mstats import gmean as geometricmean
from hnu_paper import BEAMERWIDTH, PAGEWIDTH, TEXTHEIGHT, find_latest_file#, ANG_SYS_DIR
from matplotlib import patches as mpatches
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import seaborn as sns
import arviz as av
import pandas as pd
import matplotlib.colors as mc
from matplotlib.patches import Rectangle
import colorsys
PAGEWIDTH = 6.31616
MARGINWIDTH = 1.87831
TEXTWIDTH = 4.2134 
plt.style.use("../paper.mplstyle")
module("load", "texlive/2021")
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
meta = pd.read_csv("meta.txt", delimiter="\t|\s{2,}", comment="#", header=0, engine="python")
meta = np.array(meta)   # shorten this for testing
source_names = meta[:, 0]
alert_ids = meta[:, 3]
alt_names = meta[:, 1]
```

```python
source_names, alt_names
```

```python
directories = [Path(str(_).lower().replace(" ", "_")) for _ in source_names]
```


```python
directories[0]
```

```python
params = pd.read_csv(
    "fit_params.txt", delimiter=",", header=None, comment="#"
)
log10F = np.array(params[1])
Epeak = np.array(params[2]) << u.PeV
F = np.power(10, log10F) * u.erg / u.cm**2 / u.s / 3.   # divide by three to account for only muon neutrinos (c.f. Xavier's email)

break_names = np.array(params[0])
```

```python
break_names
```

```python
energy_unit = u.erg
x_energy_unit = u.GeV
area_unit = u.cm**2
E_power = 2.
CL = [0.683, 0.95]
E_power = 2.0
```

```python
def extract_peak_contours(fit):
    try:
        E_peak = fit._fit_output["E_peak"].squeeze() << u.GeV
        peak_flux = fit._fit_output["peak_energy_flux"].squeeze() << u.GeV / u.m**2
    except:  # TODO find proper exception
        E_peak = fit._fit_output.stan_variable("E_peak").squeeze() << u.GeV
        peak_flux = (
            fit._fit_output.stan_varable("peak_energy_flux") << u.GeV / u.m**2
        )
    mask = peak_flux.value > 0.0
    data = {
        "E": E_peak.to_value(x_energy_unit, equivalencies=u.spectral())[mask],
        "flux": peak_flux.to_value(energy_unit / area_unit)[mask],
    }
    return data
```

```python
levels = np.sort(1 - np.array([0.683, 0.95]))
```

```python
light = Rectangle((0, 0), 1, 1, edgecolor="none", facecolor=lighten_color("grey", 0.25))
dark = Rectangle((0, 0), 1, 1, edgecolor="none", facecolor=lighten_color("grey", 0.6))
solid = Line2D([0], [0], ls="-", color="grey")
dashed = Line2D([0], [0], ls="--", color="grey")
```

```python
idx = np.argwhere(break_names == alt_names[0])[0][0]
print(idx)

# 1095215. uninformative
# 1100493 informative
```

```python
viper = Path("/viper/u1/jdk/icecube/hnu_paper/hnu_paper/analyses")
```

```python
# files = [_ for _ in os.listdir(viper / ) if "datafiles" in _ and "11004" in _]
# files
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
    #"4FGL J1554.2+2008"
]
```

```python
bands = (
    mpatches.Patch(facecolor=lighten_color("grey", 0.4), edgecolor="none", alpha=0.5),
    mpatches.Patch(facecolor=lighten_color("grey", 0.8), alpha=0.5, edgecolor="none")
)
```

```python
xlims = (1e3, 1e9)
ylims = {
    "4FGL J0509.4+0542": (1e-14, 2e-10),
    "4FGL J1528.4+2004": (1e-14, 2e-10),
    "4FGL J0854.0+2753": (1e-15, 2e-8),
    "4FGL J0158.8+0101": (1e-16,2e-11),
    "4FGL J0244.7+1316": (1e-16, 3e-10),
    "4FGL J1808.8+3522": (1e-15, 9e-9),
}
```

```python

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(PAGEWIDTH, PAGEWIDTH* 1.2),
                       gridspec_kw=dict(hspace=0.09, wspace=0.2))

for c, (s, ax) in enumerate(zip(sources, axs.flatten())):

    for _s, a in zip(source_names, alt_names):
        if _s == s:
            break
    
    d = s.lower().replace(" ", "_")
    d = Path(d)
    idx = np.argwhere(break_names == a)[0][0]
    scatter = ax.scatter(
        Epeak[idx].to_value(x_energy_unit, equivalencies=u.spectral()),
        F[idx].to_value(energy_unit / area_unit / u.s),
        marker="+",
        s=50,
        color="C3",
        zorder=5
    )
    if True:
        fit = StanFit.from_file(*find_latest_file(str(d / "config.yml")))
        fit_inf = StanFit.from_file(*find_latest_file(str(d / "config_am3_prior.yml")))
        # pl.plot_flux_band(E_power, CL[0], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C0", 0.6), alpha=.5,);
        #pl.plot_flux_band(E_power, CL, energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C0", 0.6), alpha=.4, );

        # fit.plot_flux_band(E_power, CL[1], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C1", 0.25), alpha=.5,);
        fit.plot_flux_band(E_power, CL[1], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C0", 0.25), alpha=.5, );
        fit.plot_flux_band(E_power, CL[0], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C0", 0.6), alpha=.5, );
        # fit.plot_peak_energy_flux(energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, linestyles="--");

        fit_inf.plot_flux_band(E_power, CL[1], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C1", 0.25), alpha=.5, );
        fit_inf.plot_flux_band(E_power, CL[0], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C1", 0.6), alpha=.5, );
        #fit_inf.plot_peak_energy_flux(energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color="C1", levels=CL, linestyles=["--", "-"]);

        

        data = extract_peak_contours(fit)
        sns.kdeplot(data, x="E", y="flux", levels=levels, ax=ax, color="C0", linestyles=["--", "-"])
        data = extract_peak_contours(fit_inf)
        sns.kdeplot(data, x="E", y="flux", levels=levels, ax=ax, color="C1", linestyles=["--", "-"])

    ax.set_xlim(1e3, 1e9)
    ax.set_ylim(*ylims[s])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator())
    ax.set_xlim(1e3, 1e9)
    ax.text(0.05, 0.95, s, va="top", ha="left", transform=ax.transAxes, fontsize=6)
    if c % 2 == 0:
        ax.set_ylabel(r"$E^2\frac{\dd[3]{n}}{\dd{E}\dd{A}\dd{t}}~[\si{\erg\per\cm\squared\per\second}]$")
    else:
        ax.set_ylabel("")
    if c >= 4:
        ax.set_xlabel(r"$E~[\si{\GeV}]$")
    # fig.savefig(d / "flux_overview.png", dpi=300, bbox_inches="tight")
    # plt.close(fig)
    # fig.savefig(d / "flux_overview_dpg.png", dpi=300, bbox_inches="tight")
    # break
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
        r"$E_\mathrm{peak}^\mathrm{obs}$---$\max{\left(E^2 \frac{\dd[3]{n}}{\dd{E}\dd{A}\dd{t}}\right)}$",
        r"95\% credibility",
        r"68\% credibility",
        r"prediction",
    ],
    loc="lower center",
    #frameon=True,
    #framealpha=0.8,
    edgecolor="none",
    ncols=3,
    bbox_to_anchor=(.5, 0.88),
    
)
```

```python
fig.savefig("flux_single_figure.png", dpi=300, bbox_inches="tight")
```

```python
figsize = ()

for a, s in zip(alt_names, source_names):
    
    
    if s not in sources:
        continue
    d = s.lower().replace(" ", "_")
    d = Path(d)
    idx = np.argwhere(break_names == a)[0][0]
    
    
    print(s, a, break_names[idx], idx)
    fig, ax = plt.subplots(figsize=(BEAMERWIDTH / 2, BEAMERWIDTH / 2 * 3 / 5))
    
    fit = StanFit.from_file(*find_latest_file("4fgl_j0509.4+0542/config.yml"))
    fit_inf = StanFit.from_file(*find_latest_file("4fgl_j0509.4+0542/config_am3_prior.yml"))
    
    pl = StanFit.from_file(*find_latest_file("pl_4fgl_j0509.4+0542/config.yml"))
    
    #pl.plot_flux_band(E_power, CL[1], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C2", 0.25), alpha=.5,);
    pl.plot_flux_band(E_power, CL[0], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C2", 0.6), alpha=.5, );

    #fit.plot_flux_band(E_power, CL[1], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C1", 0.25), alpha=.5,);
    #fit.plot_flux_band(E_power, CL[1], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C3", 0.25), alpha=.5, );
    fit.plot_flux_band(E_power, CL[0], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C3", 0.6), alpha=.5, );
    # fit.plot_peak_energy_flux(energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, linestyles="--");

    #fit_inf.plot_flux_band(E_power, CL[1], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C4", 0.25), alpha=.5, );
    fit_inf.plot_flux_band(E_power, CL[0], energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color=lighten_color("C4", 0.6), alpha=.5, );
    #fit_inf.plot_peak_energy_flux(energy_unit=energy_unit, x_energy_unit=x_energy_unit, ax=ax, color="C1", levels=CL, linestyles=["--", "-"]);
    
    scatter = ax.scatter(
        Epeak[idx].to_value(x_energy_unit, equivalencies=u.spectral()),
        F[idx].to_value(energy_unit / area_unit / u.s),
        marker="+",
        s=50,
        color="black",
        zorder=5
    )
    #data = extract_peak_contours(fit)
    #sns.kdeplot(data, x="E", y="flux", levels=levels, ax=ax, color="C0", linestyles=["--", "-"])
    #data = extract_peak_contours(fit_inf)
    #sns.kdeplot(data, x="E", y="flux", levels=levels, ax=ax, color="C1", linestyles=["--", "-"])
    #ax.set_xlim(1e3, 1e9)
    ax.legend(
        [
            #mpatches.Patch(edgecolor="none", facecolor="none"),
            bands[1],
            #mpatches.Patch(edgecolor="none", facecolor="none"),
            #mpatches.Patch(edgecolor="none", facecolor="none"),
            #dashed,
            #solid,
            scatter,
        ],
        [
            #"energy flux",
            #r"95\% credibility",
            r"68\% credibility",
            #"",
            #r"$E_\mathrm{peak}^\mathrm{obs}$---$\max{\left(E^2 \dv{n}{E}\right)}$",
            #r"95\% credibility",
            #r"68\% credibility",
            r"peak flux prediction",
        ],
        #loc="lower left",
        frameon=True,
        framealpha=0.6,
        edgecolor="none",
        #ncols=2,
        loc="lower left",
        #bbox_to_anchor=(0.5, 0.98)
    )
    ax.set_ylim(1e-14, 2e-10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e3, 1e9)
    ax.text(0.05, 0.95, s, va="top", ha="left", transform=ax.transAxes, fontsize=8)
    ax.set_ylabel(r"$E^2 \frac{\dd[3]{n}}{\dd{E}\dd{A}\dd{t}}~[\si{\erg\per\cm\squared\per\second}]$")
    ax.set_xlabel(r"$E~[\si{\GeV}]$")
    # fig.savefig(d / "flux_overview.png", dpi=300, bbox_inches="tight")
    # plt.close(fig)
    # fig.savefig(d / "flux_overview_dpg.png", dpi=300, bbox_inches="tight")
    break


```

```python

```
