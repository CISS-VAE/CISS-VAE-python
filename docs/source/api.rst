API Reference
=============

.. currentmodule:: ciss_vae

Classes
-------
.. autosummary::
   :toctree: _autosummary
   :signatures: short
   :caption: Classes in CISS-VAE

   ciss_vae.classes.vae.CISSVAE
   ciss_vae.classes.cluster_dataset.ClusterDataset
   ciss_vae.training.autotune.SearchSpace

Training & Tuning
-----------------------
.. autosummary::
   :toctree: _autosummary
   :signatures: short
   :caption: Run CISS-VAE & Automated Hyperparameter Tuning

   ciss_vae.training.autotune.autotune
   ciss_vae.utils.run_cissvae.run_cissvae

Helpers & Plotting
-------------------
.. autosummary::
   :toctree: _autosummary
   :signatures: short
   :caption: Helper functions

   ciss_vae.utils.run_cissvae.cluster_on_missing
   ciss_vae.utils.run_cissvae.cluster_on_missing_prop
   ciss_vae.utils.helpers.get_imputed_df
   ciss_vae.utils.helpers.plot_vae_architecture
   