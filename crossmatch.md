---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: hnu
    language: python
    name: python3
---

### Some notes on this notebook

Since I am using a (still) private repository for my work, I quickly tried to make this notebook self-contained.
It will download the 10 year public track dataset from harvard's dataverse to your home directory at ~/.icecube_data/
Let me know if you run into trouble executing any of the cells. Thanks again for taking the time to have a look.


```python
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from os import path
import matplotlib.pyplot as plt
import sys


from hierarchical_nu.source.parameter import Parameter

from hierarchical_nu.events import Events
from hierarchical_nu.detector.icecube import (
    IC40,
    IC59,
    IC79,
    IC86_I,
    IC86_II,
    EventType,
)
from hierarchical_nu.utils.roi import FullSkyROI, CircularROI, ROIList

from hierarchical_nu.utils.lifetime import LifeTime
from icecube_tools.utils.data import Uptime, available_data_periods
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import expon, pareto, cauchy, norm, exponnorm
from dissertation import PAGEWIDTH, TEXTWIDTH, MARGINWIDTH
plt.style.use("../paper.mplstyle")
from pastamarkers import pasta
```

```python
ROIList.clear_registry()
FullSkyROI()
Emin_det = Parameter(1e1 * u.GeV, "Emin_det")
events = Events.from_ev_file(IC40, IC59, IC79, IC86_I, IC86_II)
events.N
```

```python
mjd_min, mjd_max = events.mjd.mjd.min(), events.mjd.mjd.max()
```

### Add path to `IceCube_Gold_Bronze_Tracks.csv` from the IceCat here

```python
icecat_path = Path(path.expanduser("~/.icecube_data/IceCube_Gold_Bronze_Tracks.csv"))
icecat_path
```

```python
icecat = pd.read_csv(icecat_path)
```

```python
icecat.columns
```

```python
icecat["I3TYPE"]
```

```python
relevant_alerts = list(
    set(
        [
            "IC161103A",
            "IC161103A",
            "IC111216A",
            "IC111216A",
            "IC150831A",
            "IC170922A",
            "IC150904A",
            "IC190819A",
            "IC200107A",
            "IC190819A",
            "IC130408A",
            "IC130408A",
            "IC130408A",
            "IC151017A",
            "IC151017A",
            "IC151017A",
            "IC120515A",
            "IC110610A",
            "IC110610A",
            "IC150714A",
            "IC140114A",
            "IC140114A",
            "IC160510A",
        ]
    )
)
```

```python
def ellipse(ra, dec, ram, rap, decm, decp):
    # ellipses equation
    # x**2 / a**2 + y**2 / b**2 = 1
    # use parameterised version
    # x = a * cos(t)
    # y = b * sin(t)
    # decide which a, b from errors to use
    N = 1_000
    t = np.linspace(0, 2 * np.pi, N)
    x = np.zeros(N)
    y = np.zeros(N)
    # ra is x, dec is y, use quadrants
    x = rap * np.cos(t)
    y = decm * np.sin(t)
    x[t < 3 * np.pi / 2] = ram * np.cos(t[t < 3 * np.pi / 2])
    y[t < 3 * np.pi / 2] = decm * np.sin(t[t < 3 * np.pi / 2])

    x[t < np.pi] = ram * np.cos(t[t < np.pi])
    y[t < np.pi] = decp * np.sin(t[t < np.pi])

    x[t < np.pi / 2] = rap * np.cos(t[t < np.pi / 2])
    y[t < np.pi / 2] = decp * np.sin(t[t < np.pi / 2])
    return x + ra, y + dec


def extract_event_from_icecat(name):
    entry = icecat[icecat["NAME"] == name]
    ra = entry["RA"].iloc[0]
    dec = entry["DEC"].iloc[0]
    rap = entry["RA_ERR_PLUS"].iloc[0]
    ram = entry["RA_ERR_MINUS"].iloc[0]
    decm = entry["DEC_ERR_MINUS"].iloc[0]
    decp = entry["DEC_ERR_PLUS"].iloc[0]
    mjd = entry["EVENTMJD"].iloc[0]
    cr_veto = entry["CR_VETO"].iloc[0]
    energy = entry["ENERGY"].iloc[0]
    i3type = entry["I3TYPE"].iloc[0]
    return ra, dec, ram, rap, decm, decp, mjd, cr_veto, energy, i3type


def get_alert_id(events, mjd):
    # assoc = np.array(fit._get_event_classifications())[:, 0]  # PS component
    # get alert
    # idxs is defined over all loaded events stored in Events instance
    # search in +/- 1e-4 second time window of alert event
    idxs = np.argwhere(np.isclose(events.mjd.mjd, mjd, rtol=0.0, atol=1e-4)).flatten()
    energies = events.energies[idxs]
    if energies.size == 0:
        # no matching events found
        ret_val = np.nan
        ret_code = 2
        delta_t = np.nan
        return (ret_val, delta_t, ret_code)
    i = idxs[events.energies[idxs].argmax()]
    if idxs.size == 1:
        # found exactly one matching event
        ret_val = i
        ret_code = 1
        delta_t = mjd - events.mjd.mjd[i]
        return (ret_val, delta_t, ret_code)

    # more than one candidate
    # check if highest energy one is atleast one order of magnitude more energetic (reconstructed)
    other_energies = np.sort(energies)[:-1]
    e_ratios_bad = events.energies[i] / other_energies < 5
    if not np.any(e_ratios_bad):
        ret_code = 3
        ret_val = i
        delta_t = mjd - events.mjd.mjd[i]
        return (ret_val, delta_t, ret_code)
    else:
        ret_val = idxs
        ret_code = 4
        delta_t = mjd - events.mjd.mjd[idxs]
        return (ret_val, delta_t, ret_code)
```

```python
sep = []
names = []
track_energy = []
dt = []
sep_multi = []
names_multi = []
dt_multi = []
energy_multi = []
alert_energy = []
alert_ang_err = []
alert_mjd = []
i3type = []
pure_event_matches = []
radii = np.arange(3, 15, 1)
for name in icecat.NAME:
    ROIList.clear_registry()
    ra, dec, ram, rap, decm, decp, mjd, cr_veto, energy, i3 = extract_event_from_icecat(
        name
    )
    if mjd < mjd_min or mjd > mjd_max:
        # sort out events outside our track data set
        continue
    if cr_veto:
        # sort out events which have a cosmic ray veto flag
        continue
    # print(ra, dec, rap, ram, decm, decp, mjd)
    alert_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    for r in radii:
        ROIList.clear_registry()
        CircularROI(alert_coord, radius=r * u.deg)
        events = Events.from_ev_file(IC40, IC59, IC79, IC86_I, IC86_II)
        # print(events.N)

        track_id, delta_t, ret_code = get_alert_id(events, mjd)
        if ret_code == 2:
            continue
        track_coord = events.coords[track_id]
        if ret_code == 1 or ret_code == 3:
            sep.append(alert_coord.separation(track_coord).deg)
            track_energy.append(events.energies[track_id].to_value(u.GeV))
            dt.append(delta_t)
            alert_energy.append(energy)
            pure_event_matches.append(events[track_id])
            names.append(name)
            alert_ang_err.append(np.array([rap, ram, decm, decp]))
            i3type.append(i3)
            alert_mjd.append(mjd)
            break
        elif ret_code == 4:
            alert_mjd.append(mjd)
            sep_multi.append(alert_coord.separation(track_coord).deg)
            dt_multi.append(delta_t)
            energy_multi.append(events.energies[track_id])
            names_multi.append(name)
            break
```

```python
# assert that no multiple matching events survive
len(names_multi)
```

```python
sep = np.array(sep)
track_energy = np.array(track_energy)
```

```python
# calculate area of error ellipse
def calc_ellipse_area(ram, rap, decm, decp):
    return 1 / 4 * np.pi * (decp * ram + decm * ram + decm * rap + decp * rap)


# calculate radius of circle with same area
# I hope this does not introduce some error due to projection effects...
def calc_circularised_error(area):
    return np.sqrt(area / np.pi)
```

```python
alert_area = np.array([calc_ellipse_area(*_) for _ in alert_ang_err])
alert_circ_ang_err = np.array([calc_circularised_error(_) for _ in alert_area])
```

```python
alert_circ_ang_err.size
```

```python
ang_err = np.array([_.ang_err.to_value(u.deg) for _ in pure_event_matches])
```

```python
ids = np.zeros(len(names), dtype=bool)
for name in relevant_alerts:
    for c, all_events in enumerate(names):
        if name == all_events:
            break
    ids[c] = 1
```

```python
sum(ids)
```

```python
alert_mjd = np.array(alert_mjd)
```

```python
fig, ax = plt.subplots()
ax.scatter(np.log10(np.abs(dt)), np.log10(track_energy))
ax.scatter(np.log10(np.abs(dt))[ids], np.log10(track_energy)[ids], marker="x")
ax.set_xlabel("log10(dt)")
ax.set_ylabel("log10(Ereco/GeV)")
```

```python
dm_time_spans = []
lifetime = LifeTime()
for dm in [IC40, IC59, IC79, IC86_I, IC86_II]:
    dm_time_spans.append(lifetime.mjd_from_dm(dm))
dm_time_spans
```

```python
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(PAGEWIDTH, 4), gridspec_kw=dict(hspace=0.3, wspace=0.25))
ax = axs[0, 0]
data = ax.scatter(alert_mjd, np.abs(dt), s=2, label="all pairs")
bl = ax.scatter(alert_mjd[ids], np.abs(dt)[ids], marker="x", s=2,)
ax.set_xlabel("MJD")
ax.set_ylabel(r"$\Delta T~[\si{\second}]$")
ax.set_yscale("log")
ax.set_yticklabels(ax.get_yticklabels(), ha="left")
ax.tick_params("y", pad=22)
#for c, (times, dm) in enumerate(
#   zip(dm_time_spans, [IC40, IC59, IC79, IC86_I, IC86_II])
#):
#    ax.vlines(times, 1e-10, np.power(10, -10.5), color=f"C{c}")
#    ax.text(times[0], 2e-10, dm.P, rotation=-45, ha="left", va="bottom", rotation_mode="anchor", fontsize=6)

ax = axs[0, 1]
ax.scatter(np.sqrt(-2 * np.log(1 - 0.9)) * ang_err, alert_circ_ang_err, s=2)
ax.scatter(
    np.sqrt(-2 * np.log(1 - 0.9)) * ang_err[ids], alert_circ_ang_err[ids], marker="x", s=2
)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.plot([0, 10], [0, 10], c="black", ls=":")
ax.set_xlabel(r"track $\sigma_\mathrm{ang}$ 90\% CL [deg]")
ax.set_ylabel(r"circ.~alert $\sigma_\mathrm{ang}$ 90\% CL [deg]")
ax.set_xlim(0, xlim[1])
ax.set_ylim(0, ylim[1])


ax = axs[1, 0]
ax.scatter(np.log10(track_energy), np.log10(alert_energy) + 3, s=2)
ax.scatter(np.log10(track_energy)[ids], np.log10(alert_energy)[ids] + 3, marker="x", s=2)
ax.set_xlabel(r"$\log_{10}(\hat{E} / \si{\GeV})$")
ax.set_ylabel(r"$\log_{10}(E / \si{GeV})$")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.plot([0, 8], [0, 8], ls=":", color="black")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)

ax = axs[1, 1]
ax.hist(sep, bins=60, density=True, )
ax.vlines(
    np.array(sep)[ids], 0, 0.1, color="C1", lw=1.0
)
ax.set_xlabel("$\sphericalangle (\mathrm{alert, track})$ [deg]")
ax.set_ylabel("pdf")
ax.set_xlim(0,)
# ax.legend()
fig.legend([data, bl], [r"matched alert--track pairs", "pairs considered in this work"], loc="lower center", bbox_to_anchor=(0.5, 0.9))
# fig.savefig("/home/iwsatlas1/kuhlmann/tex/phd_thesis/figures/hnu_paper/alert_ana.png", dpi=300, bbox_inches="tight")
```

```python
fig, ax = plt.subplots()
ax.scatter(sep, np.log10(alert_energy) + 3)
ax.set_ylabel("log10(Ereco/GeV)")
ax.set_xlabel("separation [deg]")
```

```python
fig, ax = plt.subplots()
ax.scatter(np.log10(track_energy), np.log10(alert_energy) + 3)
ax.scatter(np.log10(track_energy)[ids], np.log10(alert_energy)[ids] + 3, marker="x")
ax.set_xlabel("log10(Ereco/GeV)")
ax.set_ylabel("log10(Enu/GeV)")
```

```python
fig, ax = plt.subplots()
ax.scatter(np.log10(track_energy), sep)
ax.scatter(np.log10(track_energy)[ids], np.array(sep)[ids], marker="x")
ax.set_xlabel("log10(Ereco/GeV)")
ax.set_ylabel("ang sep [deg]")
```

```python
fig, ax = plt.subplots()
ax.scatter(np.log10(alert_energy), sep)
ax.scatter(np.log10(alert_energy)[ids], np.array(sep)[ids], marker="x")
ax.set_xlabel("log10(Enu/GeV)")
ax.set_ylabel("ang sep [deg]")
```

### Fit some different functions to the distribution of opening angles

```python
exp = expon.fit(sep)
par = pareto.fit(sep)
cau = cauchy.fit(sep)
nor = norm.fit(sep)
eno = exponnorm.fit(sep)
```

```python
exp, par, cau, nor, eno
```

```python
mu = eno[1] * u.deg
sigma = eno[2] * u.deg
lam = 1 / (sigma * eno[0])
mu, sigma, lam

mu.to_value(u.rad), sigma.to_value(u.rad), lam.to_value(1 / u.rad)
mu.to_value(u.deg), sigma.to_value(u.deg), lam.to_value(1 / u.deg)
```

```python
fig, ax = plt.subplots(figsize=(MARGINWIDTH, .8 * MARGINWIDTH))
ax.hist(sep, bins=60, density=True, label="data", alpha=0.8)
ax.vlines(
   np.array(sep)[ids], 0, 0.1, color="C1", label="this work", lw=1.0
)

x = np.linspace(0, 7.2, 1_000)

#ax.plot(x, expon.pdf(x, *exp))
#ax.plot(x, pareto.pdf(x, *par))
#ax.plot(x, cauchy.pdf(x, *cau))
#ax.plot(x, norm.pdf(x, *nor))
ax.plot(x, exponnorm.pdf(x, *eno), label="fit", color="C3")
ax.set_xlabel("$\sphericalangle (\mathrm{alert, track})$ [deg]")
# ax.set_ylabel("pdf")
ax.set_yticks([])
ax.set_xlim(0, 5.1)
ax.set_xticks(np.arange(0, 6))
ax.legend()
# fig.savefig("/home/iwsatlas1/kuhlmann/tex/phd_thesis/figures/hnu_paper/ang_sys_dist_and_fit.png", dpi=300, bbox_inches="tight")
```

```python
fig, ax = plt.subplots()
ax.scatter(sep, np.sqrt(-2 * np.log(1 - 0.9)) * ang_err)
ax.scatter(sep[ids], np.sqrt(-2 * np.log(1 - 0.9)) * ang_err[ids], marker="x")
ax.set_xlabel("separation [deg]")
ax.set_ylabel("track ang err ($90\%$ CL)[deg]")
```

```python
fig, ax = plt.subplots()
ax.scatter(sep, [np.average(_) for _ in alert_ang_err])
ax.scatter(
    sep[ids],
    [np.average(alert_ang_err[i]) for i in np.argwhere(ids).flatten()],
    marker="x",
)
ax.set_xlabel("separation [deg]")
ax.set_ylabel("alert ang err ($90\%$ CL)[deg], circularised")
```

```python

```
