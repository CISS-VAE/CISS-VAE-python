def test_compute_val_mse_ignored_binary_never_reaches_bce():
    import numpy as np
    import pandas as pd
    import torch

    from ciss_vae.classes.cluster_dataset import ClusterDataset
    from ciss_vae.utils.helpers import compute_val_mse

    # ---------------------------------------
    # Create dataset
    # ---------------------------------------
    df = pd.DataFrame({
        "cont": [1.0, 2.0, 3.0, 4.0],
        "bin_ignored": [0, 1, 0, 1],
        "cat_a": [1, 0, 1, 0],
        "cat_b": [0, 1, 0, 1],
    })

    clusters = np.zeros(len(df))

    dataset = ClusterDataset(
        data=df,
        cluster_labels=clusters,
        columns_ignore=["bin_ignored"],  # 🔥 critical
        binary_feature_mask=[False, True, True, True],
        categorical_column_map={"cat": ["cat_a", "cat_b"]},
        val_proportion=1.0
    )

    # ---------------------------------------
    # Build pathological logits
    # ---------------------------------------
    true = dataset.val_data.clone()

    recon = torch.zeros_like(true)

    # continuous (fine)
    recon[:, 0] = true[:, 0]

    # EXTREME logits for ignored binary
    recon[:, 1] = torch.tensor([1000, -1000, 1000, -1000])

    # categorical logits
    recon[:, 2] = 10.0
    recon[:, 3] = -10.0

    class DummyModel:
        def __init__(self, recon):
            self.recon = recon

        def eval(self):
            pass

        def forward(self, X, C, deterministic=True):
            return self.recon, None, None

    model = DummyModel(recon)

    # ---------------------------------------
    # RUN — this used to crash
    # ---------------------------------------
    imputation_error, mse, bce, ce = compute_val_mse(
        model, dataset, debug=True
    )

    # ---------------------------------------
    # ASSERTIONS
    # ---------------------------------------

    # ✔ should not crash
    assert np.isfinite(imputation_error)

    # ✔ BCE should NOT be influenced by ignored binary
    # (since ignored binary is the only binary column)
    assert np.isclose(bce, 0.0, atol=1e-6)

    # ✔ MSE valid
    assert np.isfinite(mse)

    # ✔ CE valid
    assert np.isfinite(ce)