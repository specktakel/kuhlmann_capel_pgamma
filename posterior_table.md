### Plot table of results

```python
from hierarchical_nu.utils.config import HierarchicalNuConfig
from hierarchical_nu.utils.config_parser import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
from hnu_paper import find_latest_file
from pathlib import Path
from hierarchical_nu.fit import StanFit
from matplotlib import colors
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from hierarchical_nu.events import Events
from astropy.visualization.wcsaxes.patches import Quadrangle
from astropy.coordinates import SkyCoord
import healpy as hp
import astropy.units as u
import ligo.skymap.plot
import sys
import os
```

```python
from hnu_paper import (
    DATA_DIR,
    PAGEWIDTH,
    COLUMNWIDTH,
    BEAMERWIDTH,
    TEXTHEIGHT,
    find_latest_file,
)
from hierarchical_nu.utils.plotting import SphericalCircle
from hierarchical_nu.utils.roi import ROIList
import pandas as pd
```

```python
DATA_DIR
```

```python
plt.style.use("paper.mplstyle")
```


```python
icecat = pd.read_csv("~/.icecube_data/IceCube_Gold_Bronze_Tracks.csv")
alert_directory = Path(os.path.expanduser("~/.icecube_data/icecat/fits"))
```

```python
data_dir = Path("ptmp1/jdk/hnu_paper_data")
viper = Path("/viper") / data_dir
```

```python
alert_dict = {
    "4FGL J0158.8+0101": 55056.69828448,
    "4FGL J1528.4+2004": 55702.76655117,
    "4FGL J1533.2+1855": 55702.76655117,
    "4FGL J1554.2+2008": 55702.76655117,
    "4FGL J2030.9+1935": 55387.53618809,
    "4FGL J2030.5+2235": 55387.53618809,
    "4FGL J0955.1+3551": 58855.40437917,
}
```

```python
issued_alerts = {
    "4FGL J0158.8+0101": (29.51, 1.23, 0.38, 0.4, 0.22, 0.18),
    "4FGL J1528.4+2004": (234.94, 20.07, 0.98, 0.94, 0.93, 0.93),
    "4FGL J1533.2+1855": (234.94, 20.07, 0.98, 0.94, 0.93, 0.93),
    "4FGL J1554.2+2008": (234.94, 20.07, 0.98, 0.94, 0.93, 0.93),
    "4FGL J2030.9+1935": (307.31, 21.22, 2.5, 2.77, 1.73, 2.53),
    "4FGL J2030.5+2235": (307.31, 21.22, 2.5, 2.77, 1.73, 2.53),
    "4FGL J0955.1+3551": (148.18, 35.46, 1.83, 2.20, 1.22, 1.10),
}
```

```python

```

```python
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
    run_id = entry["RUNID"].iloc[0]
    event_id = entry["EVENTID"].iloc[0]
    return ra, dec, ram, rap, decm, decp, mjd, cr_veto, energy, run_id, event_id
```

```python
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

```

```python
meta = pd.read_csv("meta.txt", delimiter="\t|\s{2,}", comment="#", header=0, engine="python")
meta = np.array(meta)   # shorten this for testing
source_names = meta[:, 0]
alert_ids = meta[:, 3]
alt_names = meta[:, 1]
```

```python
directories = [Path(_.lower().replace(" ", "_")) for _ in source_names]
pl_directories = [Path("pl_" + _.lower().replace(" ", "_")) for _ in source_names]
```

```python
print(directories)
```

```python
files = os.listdir(alert_directory)
```

```python
pl_Nex = []
fit_Nex = []
am3_Nex = []




### table layout

# 4FGL id, 3 * assoc_prob of alert event, 3 * assoc_prob of highest scoring event, 3 * <Nex>, 3 * Pr(Nex >=1)
def compare_assoc_prob_alert():
    probs = np.zeros((len(source_names), 1 + 3 * 4), dtype="<U32")
    for c, (source, pl, d) in enumerate(zip(source_names, pl_directories, directories)):
        
        entry = 0
        probs[c, entry] = source
        
        entry += 1
        
        try: 
            pl_fit = StanFit.from_file(*find_latest_file(pl / "config.yml"))
        except:
            pl_fit = None
        try:
            # raise FileNotFoundError
            fit = StanFit.from_file(*find_latest_file(d / "config.yml"))
            #break
        except Exception as e:
            fit = None

        
        try:
            # raise FileNotFoundError
            am3 = StanFit.from_file(*find_latest_file(d / "config_am3_prior.yml"))
        except Exception as e:
            am3 = None
        print(source)
                
        if fit is None and am3 is None and pl_fit is None:
            continue
        events = pl_fit.events
        # get alert
        alert_name = alert_ids[c]

        if "IC" in alert_name:
            try:
                ra, dec, ram, rap, decm, decp, mjd, cr_veto, energy, _, _ = extract_event_from_icecat(alert_name)
            except:
                ra, dec, ram, rap, decm, decp = issued_alerts[source]
                mjd = alert_dict[source]
                cr_vet = False
                energy = np.nan
        else:
            ra, dec, ram, rap, decm, decp = issued_alerts[source]
            mjd = alert_dict[source]
            cr_vet = False
            energy = np.nan

        ret_val, delta_t, ret_code = get_alert_id(events, mjd)
        if ret_code == 4 or ret_code == 2:
            print("bad counterparts, investigate manually")
            # Nex.append(np.nan)
            # am3_Nex.append(np.nan)
            # continue
        i = ret_val
        # idxs = np.argwhere(np.isclose(events.mjd.mjd, mjd)).flatten()
        # print(events.energies[idxs])
        # print(events.mjd[idxs])
        # i = idxs[events.energies[idxs].argmax()]
        for c_f, f in enumerate([pl_fit, fit, am3]):
            if f is None:
                probs[c, 1 + c_f] = " -- "
                probs[c, 4 + c_f] = " -- "
                probs[c, 7 + c_f] = " -- "
                probs[c, 10 + c_f] = " -- "
                if c_f == 0:
                    pl_Nex.append(np.nan)
                elif c_f == 1:
                    fit_Nex.append(np.nan)
                elif c_f == 2:
                    pl_Nex.append(np.nan)
                continue    
            else:
                if not np.isnan(i):
                    assoc = np.array(f._get_event_classifications())[i, 0]  # PS component
                    probs[c, 1 + c_f] = f"{assoc:.2f}"  # alert event
                else:
                    probs[c, 1 + c_f] = " -- "
                assoc = np.array(f._get_event_classifications())[:, 0].max()  # max
                ev = np.array(f._get_event_classifications())[:, 0].argmax()
                probs[c, 4 + c_f] = f"{assoc:.2f}"
                probs[c, 7 + c_f] = f"{np.average(f['Nex_src']):.1f}"
                probs[c, 10 + c_f] = f"{np.sum(f['Nex_src'].flatten() >= 1.) / f['Nex_src'].size:.2f}"
                if ev == i and c_f == 0:
                    probs[c, 0] += r"$^{\ast}$"
                elif c_f == 0:
                    probs[c, 0] +=  "         "

                if c_f == 0:
                    pl_Nex.append(np.average(f['Nex_src']))
                elif c_f == 1:
                    fit_Nex.append(np.average(f['Nex_src']))
                elif c_f == 2:
                    am3_Nex.append(np.average(f['Nex_src']))
        # print(assoc)
        

    return probs
```

```python
probs = compare_assoc_prob_alert()
```


```python
probs
```

```python
argsort = np.argsort(am3_Nex)[::-1]
```

```python
print(" \\\ \n".join([" & ".join([_ for _ in probs[c]]) for c in range(len(probs))]))
```

```python
print(" \\\ \n".join([" & ".join([_ for _ in probs[c]]) for c in argsort]))
```

```python

```
