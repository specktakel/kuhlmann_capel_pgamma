from hierarchical_nu.utils.config import HierarchicalNuConfig
from skyllh.core.config import Config
from skyllh.datasets.i3.PublicData_10y_ps import create_dataset_collection
from skyllh.analyses.i3.publicdata_ps.time_integrated_ps import create_analysis
from skyllh.core.source_model import PointLikeSource
from skyllh.core.random import RandomStateService
from skyllh.core import tool
from iminuit import Minuit
module("load", "texlive/2021")

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
from pathlib import Path
import h5py



# Change the config to get the proper coordinates
config_file = "4fgl_j0509.4+0542/config.yml"
dir_name = Path(os.path.split(config_file)[0])
config = HierarchicalNuConfig.from_path(config_file)
config_name = os.path.splitext(os.path.split(config_file)[1])[0]
ra = config.parameter_config.src_ra[0]
dec = config.parameter_config.src_dec[0]

print(ra, dec)
cfg = Config()

dsc = create_dataset_collection(
    cfg=cfg,
    base_path="/u/jdk/.icecube_data/20210126_PS-IC40-IC86_VII/",    # this should just point to the icecube 10 year track data
    sub_path_fmt="icecube_10year_ps",
)

datasets = dsc.get_datasets(["IC40", "IC59", "IC79", "IC86_I", "IC86_II-VII"])

source = PointLikeSource(
    ra=np.deg2rad(ra), dec=np.deg2rad(dec)
)

ana = create_analysis(cfg=cfg, datasets=datasets, source=source, minimizer_impl="minuit")

events_list = [data.exp for data in ana.data_list]
ana.initialize_trial(events_list)

rss = RandomStateService(seed=1)

(log_lambda_max, fitparam_values, status) = ana.llhratio.maximize(rss)
minuit = status.minuit

fitparam_values

vars(ana.llhratio._minimizer)

minuit = status.minuit

minuit.draw_mnprofile("x0")


def neg_loglike(index, ns):
    return - ana.llhratio.evaluate([ns, index])[0]


neg_loglike(2.2, 10)

minuit = Minuit(neg_loglike, 2.2, 10)
minuit.errordef = Minuit.LIKELIHOOD
minuit.limits["ns"] = [0, 100]

minuit.migrad()
minuit.minos()

minuit.draw_profile("ns", bound=(0, 30));
plt.show()
minuit.draw_profile("index", bound=(1.0, 4.0));

# np.savetxt("txs_0506/freq_limits.dat", np.array([minuit.values[:], [minuit.merrors[0].lower, minuit.merrors[1].lower], [minuit.merrors[0].upper, minuit.merrors[1].upper]]))

minuit.merrors[1].lower

# +
minuit.draw_mncontour("ns", "index", cl=[0.68])
fig = plt.gcf()
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.vlines(minuit.params["ns"].value, *ylim)
ax.hlines(minuit.params["index"].value, *xlim)

ax.vlines(minuit.params["ns"].value + minuit.params["ns"].merror[0], *ylim)
ax.vlines(minuit.params["ns"].value + minuit.params["ns"].merror[1], *ylim)

ax.hlines(minuit.params["index"].value + minuit.params["index"].merror[0], *xlim)
ax.hlines(minuit.params["index"].value + minuit.params["index"].merror[1], *xlim)
# -

minuit.draw_profile("ns", bound=(0, 30));
fig = plt.gcf()
ax = plt.gca()
ax.hlines(1, *ax.get_xlim())

print(f"log_lambda_max = {log_lambda_max}")
print(f"fitparam_values = {fitparam_values}")
print(f"status = {status}")


def TS(x1, x2):
    return  - ana.llhratio.evaluate(np.array([x1, x2]))[0]


minuit = Minuit(TS, x1=10 , x2=2.5)
minuit.errordef = Minuit.LIKELIHOOD

minuit.migrad()

minuit.draw_mnmatrix()

minuit.draw_profile("x1")

TS = ana.calculate_test_statistic(log_lambda_max, fitparam_values)
print(f"TS = {TS:.3f}")


rss = RandomStateService(seed=1)


(ts, x, status) = ana.unblind(rss)

print(f'ns = {x["ns"]:.2f}')
print(f'gamma = {x["gamma"]:.2f}')


(llhratio_value, (grad_ns, grad_gamma)) = ana.llhratio.evaluate([x["ns"], x["gamma"]])
print(f"llhratio_value = {llhratio_value:.3f}")
print(f"grad_ns = {grad_ns:.3f}")
print(f"grad_gamma = {grad_gamma:.3f}")

(ns_min, ns_max, ns_step) = (0, 90, 0.5)
(gamma_min, gamma_max, gamma_step) = (1.5, 4.1, 0.1)

ns_edges = np.linspace(ns_min, ns_max, int((ns_max - ns_min) / ns_step) + 1)
ns_vals = 0.5 * (ns_edges[1:] + ns_edges[:-1])

gamma_edges = np.linspace(
    gamma_min, gamma_max, int((gamma_max - gamma_min) / gamma_step + 1)
)
gamma_vals = 0.5 * (gamma_edges[1:] + gamma_edges[:-1])

delta_ts = np.empty((len(ns_vals), len(gamma_vals)), dtype=np.double)
for ns_i, ns in enumerate(ns_vals):
    for gamma_i, gamma in enumerate(gamma_vals):

        delta_ts[ns_i, gamma_i] = ana.calculate_test_statistic(
            llhratio_value, [x["ns"], x["gamma"]]
        ) - ana.calculate_test_statistic(
            ana.llhratio.evaluate([ns, gamma])[0], [ns, gamma]
        )

x["gamma"]

ana.llhratio.evaluate([x["ns"], x["gamma"]])[0] - ana.llhratio.evaluate([x["ns"], x["gamma"] + 0.3])[0]


index_max = np.argmin(delta_ts)
ns_i_max = int(index_max / len(gamma_vals))
gamma_i_max = index_max % len(gamma_vals)
ns_best = ns_vals[ns_i_max]
gamma_best = gamma_vals[gamma_i_max]

with h5py.File(dir_name / f"{config_name}_skyllh.h5", "w") as f:
    f.create_dataset("gamma_vals", data=gamma_vals)
    f.create_dataset("ns_vals", data=ns_vals)
    f.create_dataset("delta_ts", data=delta_ts)
# -

vars(ana.llhratio.minimizer)






# Determine the delta lambda value for the 95% quantile assuming a chi-sqaure
# distribution with 2 degrees of freedom (i.e. assuming Wilks theorem).
chi2_68_quantile = scipy.stats.chi2.ppf(0.68, df=2)
chi2_90_quantile = scipy.stats.chi2.ppf(0.90, df=2)
chi2_95_quantile = scipy.stats.chi2.ppf(0.95, df=2)


### Do this here to extract the likelihood contours at the specified confidence levels

plt.figure(figsize=(8, 6))
plt.pcolormesh(gamma_edges, ns_edges, delta_ts, cmap="nipy_spectral")
cbar = plt.colorbar()
cbar.set_label(r"$\Delta$TS")
contour_68 = plt.contour(
    gamma_vals, ns_vals, delta_ts, [chi2_68_quantile], colors="#FFFFFF"
)
contour_90 = plt.contour(
    gamma_vals, ns_vals, delta_ts, [chi2_90_quantile], colors="#AAAAAA"
)
contour_95 = plt.contour(
    gamma_vals, ns_vals, delta_ts, [chi2_95_quantile], colors="#444444"
)
plt.plot(gamma_best, ns_best, marker="x", color="white", ms=10)
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$n_{\mathrm{s}}$")
plt.ylim(ns_min, ns_max)
plt.xlim(gamma_min, gamma_max)

one_sigma_limits = np.vstack(
    (contour_68._paths[0].get_extents()._points[:, 0], contour_68._paths[0].get_extents()._points[:, 1])
).T

output = np.vstack((fitparam_values[::-1], one_sigma_limits))

outfile = dir_name / Path("freq_limits.dat")

np.savetxt(outfile, output)
