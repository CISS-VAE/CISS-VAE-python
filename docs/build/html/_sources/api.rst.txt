API Reference
=============

.. currentmodule:: ciss_vae
.. autosummary::
   :toctree: _autosummary
   :recursive:

   run_cissvae

.. currentmodule:: ciss_vae.classes
.. autosummary::
   :toctree: _autosummary
   :recursive:

   vae.CISSVAE
   cluster_dataset.ClusterDataset

.. currentmodule:: ciss_vae.training
.. autosummary::
   :toctree: _autosummary
   :recursive:

   autotune.SearchSpace
   autotune.autotune
   train_initial.train_vae_initial
   train_refit.impute_and_refit_loop

.. currentmodule:: ciss_vae.utils
.. autosummary::
   :toctree: _autosummary
   :recursive:

   run_cissvae.run_cissvae
   run_cissvae.cluster_on_missing
   run_cissvae.cluster_on_missing_prop
   helpers.get_imputed_df
   helpers.plot_vae_architecture
