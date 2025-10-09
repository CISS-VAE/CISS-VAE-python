---
title: CISS-VAE Quickstart
toc-title: Table of contents
---

The **Clustering-Informed Shared-Structure Variational Autoencoder (CISS-VAE)** is a flexible deep learning model for missing data imputation that accommodates all three types of missing data mechanisms: Missing Completely At Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). While it is particularly well-suited to MNAR scenarios where missingness patterns carry informative signals, CISS-VAE also functions effectively under MAR assumptions.

# Installation
The CISS-VAE package is currently available for python, with an R
package to be released soon. It can be installed from either
[github](https://github.com/CISS-VAE/CISS-VAE-python) or PyPI.

``` bash
# From PyPI (not released yet)
pip install ciss-vae

```

``` bash
# From GitHub (latest development version)
pip install git+https://github.com/CISS-VAE/CISS-VAE-python.git
```

<div>

> **Note**
>
> If you want run_cissvae to handle clustering, please install the
> clustering dependencies scikit-learn, leidenalg, python-igraph with pip.
>
> ``` bash
> pip install scikit-learn leidenalg python-igraph
>
> OR
>
> pip install ciss-vae[clustering]
> ```

</div>

# Quickstart

To load the sample dataset:

``` python
from ciss_vae.data import load_example_dataset
df_missing, df_complete, clusters = load_example_dataset()
```

To run the CISSVAE imputation model with default parameters (assuming known clusters).

``` python
import pandas as pd
from ciss_vae.training.run_cissvae import run_cissvae
from ciss_vae.data import load_example_dataset

# optional, display vae architecture
from ciss_vae.utils.helpers import plot_vae_architecture

df_missing, _, clusters = load_example_dataset()

imputed_data, vae = run_cissvae(
    data = data,
    columns_ignore = data.columns[:5], ## columns to ignore when selecting validation dataset (and clustering if you do not provide clusters).
    clusters = clusters
)

## OPTIONAL - PLOT VAE ARCHITECTURE
plot_vae_architecture(model = vae,
                        title = None)
```
![Output of plot_vae_architecture](image-1v2.png)

To have run_cissvae() perform data clustering with Leiden: 

``` python
import pandas as pd
from ciss_vae.training.run_cissvae import run_cissvae
from ciss_vae.data import load_example_dataset

df_missing, _, _ = load_example_dataset()

imputed_data, vae = run_cissvae(
    data = data,
    columns_ignore = data.columns[:5], ## columns to ignore when selecting validation dataset (and clustering if you do not provide clusters).
    clusters = None,
    leiden_resolution = 0.005
)
```