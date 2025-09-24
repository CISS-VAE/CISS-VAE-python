.. CISS-VAE documentation master file, created by
   sphinx-quickstart on Mon Jul 21 09:06:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CISS-VAE documentation
======================

The Clustering-Informed Shared-Structure Variational Autoencoder (CISS-VAE) is a flexible deep 
learning model for missing data imputation that is particularly well-suited to MNAR (Missing Not at Random) scenarios where missingness patterns are informative.
It also functions effectively under MAR (Missing at Random) assumptions. 
 
The model uses unsupervised clustering to capture distinct patterns of missingness and leverages a mix of shared and unshared encoder 
and decoder layers, allowing knowledge transfer across clusters and enhancing parameter stability. Its iterative learning procedure improves imputation 
accuracy compared to traditional training approaches.  
 
The CISS-VAE package also offers the :py:func:`ciss_vae.training.autotune.autotune` function, which can help select the best hyperparameters for your model within a user-defined search space. 
The autotune function has compatibility with Optuna Dashboard for viewing hyperparameter importance trends. 

The R package associated with this model can be found at `rCISS-VAE <https://ciss-vae.github.io/rCISS-VAE/>`_. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   vignette
   missingness_prop_vignette
   dni_vignette.ipynb
   api


Installation
------------

To install via PyPI:

.. code-block:: bash

   pip install ciss-vae


To install via github: 
.. code-block:: bash

   pip install git+https://github.com/CISS-VAE/CISS-VAE-python.git


Features
--------

- Cluster-specific VAE architecture
- Compatible with real-world missing data (MAR, MNAR)
- Optuna-based hyperparameter tuning


Need help? See the `vignette <vignette.html>`_ or the full `API reference <api.html>`_.
