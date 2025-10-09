import pytest
import numpy as np
import pandas as pd
import torch
from ciss_vae.training.run_cissvae import run_cissvae
from ciss_vae.classes.vae import CISSVAE
from ciss_vae.classes.cluster_dataset import ClusterDataset


def test_dni_excludes_validation_cells(small_df, small_dni, one_cluster_labels):
    """
    DNI-marked cells must NEVER be selected as validation targets.
    That means those positions are NaN in ds.val_data.
    """
    ds = ClusterDataset(
        data=small_df,                    # (N, P) features only (index='id')
        cluster_labels=one_cluster_labels,
        val_proportion=0.5,              # force some validation holdout
        replacement_value=0.0,
        columns_ignore=["id"],               # all columns eligible unless DNI says otherwise
        imputable=small_dni,
    )

    val_data = ds.val_data.numpy()
    dni_np = ~small_dni.values.astype(bool)
    assert np.all(np.isnan(val_data[dni_np])), (
        "DNI-marked cells should remain NaN in val_data (never selected for validation b/c they are missing in the first place)."
    )


def test_dni_cells_remain_nan_in_imputed_dataset(small_df, small_dni, tiny_train_kwargs):
    """
    End-to-end: cells with NaN & DNI==True must remain NaN in final imputed output.
    """
    # run_cissvae expects a column to use as an index; give it explicitly

    res = run_cissvae(
        data=small_df,
        columns_ignore=[],                # features are exactly small_df.columns
        imputable_matrix=small_dni,   # same shape & col names as features
        **tiny_train_kwargs,
    )

    # Align output to the original feature frame for assertions
    imp = res
    assert isinstance(imp, pd.DataFrame)

    # The two injected NaNs protected by DNI must remain NaN
    assert np.isnan(imp.iloc[0, 0]), "DNI-protected missing cell (0,0) must remain NaN."
    assert np.isnan(imp.iloc[1, 1]), "DNI-protected missing cell (1,1) must remain NaN."


def test_run_cissvae_accepts_dni_and_returns_expected_shapes(small_df, small_dni):
    """
    Smoke + shape test: run_cissvae accepts DNI and returns a sensible frame.
    """

    res = run_cissvae(
        data=small_df,
        columns_ignore=[],
        imputable_matrix=small_dni,
        val_proportion=0.1,
        epochs=1,
        max_loops=1,
        patience=1,
        k_neighbors = 5,
        return_model=True,
        return_clusters=True,
        return_history=False,
        return_dataset=False,
        verbose=False,
    )

    imp = res[0]
    assert isinstance(imp, pd.DataFrame)
    # same # rows as input
    n_in = small_df.shape[0]
    assert imp.shape[0] == n_in

    # The imputed frame should at least include the original feature columns
    # if "id" in imp.columns:
    #     imp = imp.set_index("id")
    assert set(small_df.columns).issubset(set(imp.columns))


def test_dni_wrong_shape_raises(small_df, small_dni, one_cluster_labels):
    """
    If DNI shape mis-matches data shape, ClusterDataset should raise a clear error.
    """
    # remove a column -> shape mismatch
    bad_dni = small_dni.iloc[:, :-1]

    with pytest.raises((ValueError, AssertionError, TypeError)):
        _ = ClusterDataset(
            data=small_df,
            cluster_labels=one_cluster_labels,
            val_proportion=0.1,
            columns_ignore=[],
            imputable=bad_dni,
        )