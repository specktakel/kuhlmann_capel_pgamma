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
from matplotlib import pyplot as plt
from hierarchical_nu.source.flux_model import LogParabolaSpectrum, PowerLawSpectrum
from hierarchical_nu.source.parameter import Parameter
import numpy as np
import astropy.units as u
from hnu_paper import COLUMNWIDTH
from matplotlib.lines import Line2D

module("load", "texlive/2021")
# plt.style.use("../paper.mplstyle")
```

```python
data = np.loadtxt("rodrigues_spectrum_nu_F_nu.csv", delimiter=",")
```

```python
freq = np.power(10, data[:, 0]) << u.Hz
E = freq.to(u.GeV, equivalencies=u.spectral())
```

```python
nu_F = np.power(10, data[:, 1]) << u.erg / u.cm**2 / u.s
```

```python
F = (nu_F / E**2).to(1 / u.GeV / u.cm**2 / u.s)
```

```python
Parameter.clear_registry()
norm = Parameter(F.max()*0.98, "norm")
norm_energy = Parameter(1.5e4*u.GeV, "norm_energy")
alpha = Parameter(0., "src_index")
beta = Parameter(0.7, "beta_index")
logparabola = LogParabolaSpectrum(norm, norm_energy, alpha, beta)

index = Parameter(0., "index")
pl_norm = Parameter(1e-10 / u.GeV / u.s / u.cm**2, "pl_norm")

pl = PowerLawSpectrum(pl_norm, 1e4 * u.GeV, index)
```

```python
def logparabola(E, N0, alpha, beta, E0):
    return N0 * np.power(E/E0, -alpha - beta * np.log(E/E0))

def powerlaw(E, N0, alpha, E0):
    return N0 * np.power(E/E0, -alpha)

def mixed(E, N0_lp, alpha_lp, beta_lp, E0_lp, alpha_pl):
    E = np.atleast_1d(E)
    Ebreak = np.exp((-alpha_pl + alpha_lp) / (-beta_lp)) * E0_lp
    print(Ebreak)
    norm = logparabola(Ebreak, N0_lp, alpha_lp, beta_lp, E0_lp)
    output = np.zeros(E.shape)
    output[E>=Ebreak] = logparabola(E[E>=Ebreak], N0_lp, alpha_lp, beta_lp, E0_lp)
    output[E<Ebreak] = powerlaw(E[E<Ebreak], norm, alpha_pl, Ebreak)
    return output
```

```python
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(MARGINWIDTH, 2.5), sharex=True, gridspec_kw={"height_ratios": [5, 1], "hspace": 0.05})

axs[0].plot(E.to_value(u.GeV), E.to_value(u.GeV)**2 * F.to_value(1 / u.GeV / u.s / u.cm**2), label="R+")

blue = Line2D([0], [0], color="C0")
orange = Line2D([0], [0], color="C1")

handles = [blue, orange,]
labels = [r"$p\gamma$", r"\texttt{hnu} approximation"]

fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.9))

# Overplot logparabola spectrum
x = np.geomspace(E.to_value(u.GeV).min(), E.to_value(u.GeV).max(), 1_000) << u.GeV
flux = logparabola(x, norm.value, alpha.value, beta.value, norm_energy.value)



flux = pl(x)
axs[0].plot(x.to_value(u.GeV), x.to_value(u.GeV)**2 * mixed(x.to_value(u.GeV), norm.value.value, alpha.value, beta.value, norm_energy.value.value, index.value), label="approximation")


axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_ylabel(r"$E^2\frac{\dd[3]{n}}{\dd{E}\dd{A}\dd{t}}$~[\si{\GeV\per\cm\squared\per\second}]")

axs[1].plot(
    E.to_value(u.GeV),
    F.to_value(1 / u.GeV / u.s / u.cm**2) / mixed(E.to_value(u.GeV), norm.value.value, alpha.value, beta.value, norm_energy.value.value, index.value),
    color="black",
    #label="am3 / approx"
)
axs[1].set_xlabel(r"$E~[\si{\GeV}]$")
axs[1].set_ylabel(r"$\frac{p\gamma}{\mathrm{approx.}}$")
# fig.savefig("logparabola_approx_E2dNdE.png", bbox_inches="tight", dpi=300)
```

```python

```
