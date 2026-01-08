from hierarchical_nu.utils.config import HierarchicalNuConfig
from skyllh.core.config import Config
from skyllh.datasets.i3.PublicData_10y_ps import create_dataset_collection
from skyllh.analyses.i3.publicdata_ps.time_integrated_ps import create_analysis
from skyllh.core.source_model import PointLikeSource
from skyllh.core.random import RandomStateService

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
from pathlib import Path
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

config_file = args.config
dir_name = Path(os.path.split(args.config)[0])
config = HierarchicalNuConfig.from_path(config_file)
config_name = os.path.splitext(os.path.split(config_file)[1])[0]
ra = config.parameter_config.src_ra[0]
dec = config.parameter_config.src_dec[0]

print(ra, dec)
cfg = Config()

dsc = create_dataset_collection(
    cfg=cfg,
    base_path="/u/jdk/.icecube_data/20210126_PS-IC40-IC86_VII/",
    sub_path_fmt="icecube_10year_ps",
)

datasets = dsc.get_datasets(["IC40", "IC59", "IC79", "IC86_I", "IC86_II-VII"])

source = PointLikeSource(
    ra=np.deg2rad(ra), dec=np.deg2rad(dec)
)

ana = create_analysis(cfg=cfg, datasets=datasets, source=source)

events_list = [data.exp for data in ana.data_list]
ana.initialize_trial(events_list)

rss = RandomStateService(seed=1)

(log_lambda_max, fitparam_values, status) = ana.llhratio.maximize(rss)

print(f"log_lambda_max = {log_lambda_max}")
print(f"fitparam_values = {fitparam_values}")
print(f"status = {status}")

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

# Determine the best fit ns and gamma values from the scan.
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

# Determine the delta lambda value for the 95% quantile assuming a chi-sqaure
# distribution with 2 degrees of freedom (i.e. assuming Wilks theorem).
chi2_68_quantile = scipy.stats.chi2.ppf(0.68, df=2)
chi2_90_quantile = scipy.stats.chi2.ppf(0.90, df=2)
chi2_95_quantile = scipy.stats.chi2.ppf(0.95, df=2)


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