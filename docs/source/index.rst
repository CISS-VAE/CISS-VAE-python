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
 
The CISS-VAE package also offers an :py:func:`ciss_vae.training.autotune.autotune` function that will select the best hyperparameters for your model within a user-defined search space, and compatibility with Optuna Dashboard for viewing hyperparameter importance trends. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   vignette
   api


Installation
------------

.. code-block:: bash

   pip install ciss-vae


Features
--------

- Cluster-specific VAE architecture
- Compatible with real-world missing data
- Optuna-based hyperparameter tuning
- Clean API for training and imputation


Need help? See the `vignette <vignette.html>`_ or the full `API reference <api.html>`_.
