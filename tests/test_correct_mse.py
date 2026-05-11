def test_compute_val_mse_correctness_all_losses():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from ciss_vae.classes.cluster_dataset import ClusterDataset
    from ciss_vae.utils.helpers import compute_val_mse

    torch.manual_seed(0)
    np.random.seed(0)

    # ---------------------------------------
    # Construct dataset (all types)
    # ---------------------------------------
    df = pd.DataFrame({
        # continuous
        "cont": [1.0, 2.0],

        # binary
        "bin": [0.0, 1.0],

        # categorical (one-hot)
        "cat_a": [1.0, 0.0],
        "cat_b": [0.0, 1.0],
    })

    clusters = np.zeros(len(df))

    dataset = ClusterDataset(
        data=df,
        cluster_labels=clusters,
        columns_ignore=None,
        binary_feature_mask=[False, True, True, True],
        categorical_column_map={"cat": ["cat_a", "cat_b"]},
        val_proportion=1.0
    )

    # ---------------------------------------
    # Build recon logits manually
    # ---------------------------------------
    true = dataset.val_data.clone()

    recon = torch.zeros_like(true)

    # continuous (slightly wrong)
    recon[:, 0] = true[:, 0] + 1.0

    # binary logits
    # sigmoid(0) = 0.5
    recon[:, 1] = torch.tensor([0.0, 0.0])

    # categorical logits
    # row 1 → correct class
    # row 2 → wrong class
    recon[:, 2] = torch.tensor([10.0, -10.0])
    recon[:, 3] = torch.tensor([-10.0, 10.0])

    class DummyModel:
        def __init__(self, recon):
            self.recon = recon

        def eval(self):
            pass

        def forward(self, X, C, deterministic=True):
            return self.recon, None, None

    model = DummyModel(recon)

    # ---------------------------------------
    # Run function
    # ---------------------------------------
    total, mse, bce, ce = compute_val_mse(model, dataset)

    # ---------------------------------------
    # MANUAL CALCULATIONS
    # ---------------------------------------

    # ---- MSE ----
    pred_cont = recon[:, 0]
    true_cont = true[:, 0]
    expected_mse = torch.mean((pred_cont - true_cont) ** 2).item()

    # ---- BCE ----
    logits = recon[:, 1]
    targets = true[:, 1]
    expected_bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="mean"
    ).item()

    # ---- CE ----
    logits_cat = recon[:, 2:4]
    target_cat = torch.argmax(true[:, 2:4], dim=1)
    expected_ce = F.cross_entropy(logits_cat, target_cat, reduction="mean").item()

    # ---------------------------------------
    # ASSERTIONS
    # ---------------------------------------

    assert np.isclose(mse, expected_mse, atol=1e-6), f"MSE mismatch: {mse} vs {expected_mse}"
    assert np.isclose(bce, expected_bce, atol=1e-6), f"BCE mismatch: {bce} vs {expected_bce}"
    assert np.isclose(ce, expected_ce, atol=1e-6), f"CE mismatch: {ce} vs {expected_ce}"

    # total check
    assert np.isclose(total, mse + bce + ce, atol=1e-6)