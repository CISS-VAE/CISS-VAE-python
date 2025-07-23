# pyCISS-VAE

Python implementation of the Clustering-Informed Shared-Structure Variational Autoencoder (CISS-VAE). 

CISS-VAE is a flexible deep learning model for missing data imputation that accommodates all three types of missing data mechanisms: Missing Completely At Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). While it is particularly well-suited to MNAR scenarios where missingness patterns carry informative signals, CISS-VAE also functions effectively under MAR assumptions.

## Installation

The CISS-VAE package is currently available for python, with an R
package to be released soon. It can be installed from either
[github](https://github.com/CISS-VAE/CISS-VAE-python) or PyPI.

``` bash
# From PyPI
pip install ciss-vae

```
OR

``` bash
# From GitHub (latest development version)
pip install git+https://github.com/CISS-VAE/CISS-VAE-python.git
```

<div>

> **Note**
>
> If you want run_cissvae to handle clustering, please install the
> clustering dependencies scikit-learn and hdbscan with pip.
>
> ``` bash
> pip install scikit-learn hdbscan
>
> OR
>
> pip install ciss-vae[clustering]
> ```

</div>``

## Quickstart Tutorial

The full vignette can be found [here](https://ciss-vae-python.readthedocs.io/en/latest/vignette.html#).


