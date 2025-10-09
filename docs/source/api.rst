API Reference
=============

.. currentmodule:: ciss_vae

Classes
---------
.. currentmodule:: ciss_vae.classes
.. autosummary::
   :toctree: _autosummary
   :recursive:

   vae.CISSVAE
   cluster_dataset.ClusterDataset

Training & Tuning
-------------------
.. currentmodule:: ciss_vae.training
.. autosummary::
   :toctree: _autosummary
   :recursive:

   autotune.SearchSpace
   autotune.autotune
   run_cissvae.run_cissvae

Utilities & Helpers
---------------------
.. currentmodule:: ciss_vae.utils
.. autosummary::
   :toctree: _autosummary
   :recursive:

   
   clustering.cluster_on_missing
   clustering.cluster_on_missing_prop
   matrix.create_missingness_prop_matrix
   helpers.plot_vae_architecture
   helpers.get_imputed_df
