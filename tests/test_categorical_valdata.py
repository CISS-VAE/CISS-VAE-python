import pytest
import numpy as np
import pandas as pd
import torch
from ciss_vae.classes.cluster_dataset import ClusterDataset
from ciss_vae.training.run_cissvae import run_cissvae
from ciss_vae.utils.helpers import compute_val_mse


class TestClusterDatasetCategoricalValidation:
    """
    Test suite for grouped categorical validation masking behavior in
    ClusterDataset.

    These tests verify that:
    1. validation units use original categorical names
    2. all dummy columns from the same original categorical are masked together
    3. non-categorical columns remain independent
    4. per-cluster validation proportions still work correctly
    """

    def _build_dataset(
        self,
        *,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
        data=None,
        cluster_labels=None,
        val_proportion=0.5,
        val_seed=123,
        categorical_column_map=None,
        binary_feature_mask=None,
        columns_ignore=None,
        replacement_value=0.0,
    ):
        """
        Convenience builder for ClusterDataset used across tests.

        Parameters default to the conftest fixtures so that tests remain short
        and readable, while still allowing per-test overrides.
        """
        if data is None:
            data = categorical_validation_base_dataframe.copy()

        if cluster_labels is None:
            cluster_labels = categorical_validation_cluster_labels.copy()

        if categorical_column_map is None:
            categorical_column_map = categorical_validation_column_map

        if binary_feature_mask is None:
            binary_feature_mask = categorical_validation_binary_feature_mask.copy()

        

        return ClusterDataset(
            data=data,
            cluster_labels=cluster_labels,
            val_proportion=val_proportion,
            replacement_value=replacement_value,
            columns_ignore=columns_ignore,
            val_seed=val_seed,
            binary_feature_mask=binary_feature_mask,
            categorical_column_map=categorical_column_map,
        )

    def test_validation_units_use_original_category_names(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        The new behavior should create validation units keyed by original
        categorical names ("C1", "C2"), not by individual dummy variables.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
        )

        expected_unit_names = {"X1", "X2", "B1", "B2", "C1", "C2"}
        assert set(ds.validation_units.keys()) == expected_unit_names

        assert ds.validation_units["C1"]["kind"] == "categorical"
        assert ds.validation_units["C2"]["kind"] == "categorical"

        c1_cols = [ds.feature_names[i] for i in ds.validation_units["C1"]["cols"]]
        c2_cols = [ds.feature_names[i] for i in ds.validation_units["C2"]["cols"]]

        assert c1_cols == ["C1b1", "C1b2"]
        assert c2_cols == ["C2b1", "C2b2"]

        assert "C1b1" not in ds.validation_units
        assert "C1b2" not in ds.validation_units
        assert "C2b1" not in ds.validation_units
        assert "C2b2" not in ds.validation_units

    def test_non_categorical_columns_remain_single_column_units(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        Continuous and standalone binary columns should still be ordinary
        one-column validation units.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
        )

        for feature_name in ["X1", "X2", "B1", "B2"]:
            assert ds.validation_units[feature_name]["kind"] in ["binary", "continuous"]
            cols = ds.validation_units[feature_name]["cols"]
            assert len(cols) == 1
            assert ds.feature_names[cols[0]] == feature_name

    def test_grouped_categorical_masking_masks_all_dummy_columns_together(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        If one dummy column in a grouped categorical is selected for validation
        in a row, all dummy columns in that categorical must be selected.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=999,
        )

        val_mask = ds.val_mask.cpu().numpy()

        c1_mask_1 = val_mask[:, ds.feature_names.index("C1b1")]
        c1_mask_2 = val_mask[:, ds.feature_names.index("C1b2")]
        c2_mask_1 = val_mask[:, ds.feature_names.index("C2b1")]
        c2_mask_2 = val_mask[:, ds.feature_names.index("C2b2")]

        assert np.array_equal(c1_mask_1, c1_mask_2)
        assert np.array_equal(c2_mask_1, c2_mask_2)

    def test_grouped_categorical_masking_selects_expected_number_of_rows_per_category(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        With 4 rows and val_proportion=0.5, each grouped categorical should
        select exactly 2 rows. Since each category has 2 dummy columns, that
        gives 4 held-out cells per grouped categorical.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=123,
        )

        val_mask = ds.val_mask.cpu().numpy()

        c1_cols = [ds.feature_names.index("C1b1"), ds.feature_names.index("C1b2")]
        c2_cols = [ds.feature_names.index("C2b1"), ds.feature_names.index("C2b2")]

        c1_row_selected = np.any(val_mask[:, c1_cols], axis=1)
        c2_row_selected = np.any(val_mask[:, c2_cols], axis=1)

        assert int(c1_row_selected.sum()) == 2
        assert int(c2_row_selected.sum()) == 2

        assert int(val_mask[:, c1_cols].sum()) == 4
        assert int(val_mask[:, c2_cols].sum()) == 4

    def test_val_data_contains_only_validation_entries_for_grouped_categoricals(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        val_data should equal raw_data where val_mask is True and be NaN
        everywhere else.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=321,
        )

        raw_np = ds.raw_data.cpu().numpy()
        val_data_np = ds.val_data.cpu().numpy()
        val_mask_np = ds.val_mask.cpu().numpy()

        assert np.allclose(
            val_data_np[val_mask_np],
            raw_np[val_mask_np],
            equal_nan=True,
        )
        assert np.all(np.isnan(val_data_np[~val_mask_np]))

    def test_masks_are_inverse_of_val_mask_when_input_has_no_original_missingness(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        In the fully observed toy dataset, the observed training mask should be
        exactly the inverse of val_mask after validation selection.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=222,
        )

        val_mask_np = ds.val_mask.cpu().numpy()
        observed_mask_np = ds.masks.cpu().numpy()

        assert np.array_equal(observed_mask_np, ~val_mask_np)

    def test_grouped_masking_is_reproducible_for_same_seed(
    self,
    categorical_validation_base_dataframe,
    categorical_validation_cluster_labels,
    categorical_validation_column_map,
    categorical_validation_binary_feature_mask,
    ):
        """
        Same seed and same data should produce identical grouped validation
        masking.
        """
        ds1 = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=555,
        )
        ds2 = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=555,
        )

        # Validation mask should match exactly.
        assert torch.equal(ds1.val_mask, ds2.val_mask)

        # val_data contains NaNs in non-validation positions, so use a NaN-aware comparison.
        assert torch.allclose(ds1.val_data, ds2.val_data, equal_nan=True)

        # Observed training masks should also match exactly.
        assert torch.equal(ds1.masks, ds2.masks)

    def test_grouped_masking_changes_with_different_seed(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        Different seeds should typically produce different grouped validation
        masks.
        """
        ds1 = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=111,
        )
        ds2 = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=222,
        )

        assert not torch.equal(ds1.val_mask, ds2.val_mask)

    def test_grouped_categorical_requires_all_dummy_columns_observed_for_candidate_row(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        A row should be eligible for grouped categorical holdout only when all
        dummy variables for that grouped categorical are observed.
        """
        data = categorical_validation_base_dataframe.copy()
        data.loc[0, "C1b1"] = np.nan
        data.loc[0, "C1b2"] = 0.0

        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            data=data,
            val_proportion=0.5,
            val_seed=123,
        )

        val_mask = ds.val_mask.cpu().numpy()
        c1b1_idx = ds.feature_names.index("C1b1")
        c1b2_idx = ds.feature_names.index("C1b2")

        assert not val_mask[0, c1b1_idx]
        assert not val_mask[0, c1b2_idx]

    def test_grouped_categorical_with_one_eligible_row_masks_that_row_when_prop_positive(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        If only one row is eligible for a grouped categorical and the holdout
        proportion is positive, that row should be selected.
        """
        data = categorical_validation_base_dataframe.copy()

        data.loc[0, ["C1b1", "C1b2"]] = [np.nan, 0.0]
        data.loc[1, ["C1b1", "C1b2"]] = [1.0, np.nan]
        data.loc[2, ["C1b1", "C1b2"]] = [np.nan, np.nan]
        data.loc[3, ["C1b1", "C1b2"]] = [0.0, 1.0]

        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            data=data,
            val_proportion=0.1,
            val_seed=123,
        )

        val_mask = ds.val_mask.cpu().numpy()
        c1_cols = [ds.feature_names.index("C1b1"), ds.feature_names.index("C1b2")]

        c1_row_selected = np.any(val_mask[:, c1_cols], axis=1)
        expected = np.array([False, False, False, True])

        assert np.array_equal(c1_row_selected, expected)

    def test_non_grouped_columns_are_sampled_independently(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        Ordinary columns should still be sampled independently and should not
        share forced mask patterns.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=42,
        )

        val_mask = ds.val_mask.cpu().numpy()
        x1 = val_mask[:, ds.feature_names.index("X1")]
        x2 = val_mask[:, ds.feature_names.index("X2")]

        assert not np.array_equal(x1, x2)

    def test_standalone_binary_columns_are_not_grouped_unless_in_categorical_map(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        Standalone binary columns like B1 and B2 should remain independent
        validation units unless explicitly grouped.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=84,
        )

        val_mask = ds.val_mask.cpu().numpy()
        b1 = val_mask[:, ds.feature_names.index("B1")]
        b2 = val_mask[:, ds.feature_names.index("B2")]

        assert not np.array_equal(b1, b2)

    def test_grouped_categorical_masking_respects_cluster_specific_proportions(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        Validation masking should still be done separately within each cluster.

        With two clusters of four rows each and proportions:
            cluster 0 -> 0.25 -> 1 selected row
            cluster 1 -> 0.50 -> 2 selected rows

        grouped categorical C1 should follow those counts.
        """
        data = pd.concat(
            [categorical_validation_base_dataframe, categorical_validation_base_dataframe],
            ignore_index=True,
        )
        cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=np.zeros(len(categorical_validation_base_dataframe), dtype=int),
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            data=data,
            cluster_labels=cluster_labels,
            val_proportion={0: 0.25, 1: 0.50},
            val_seed=123,
        )

        val_mask = ds.val_mask.cpu().numpy()
        c1_cols = [ds.feature_names.index("C1b1"), ds.feature_names.index("C1b2")]

        c1_selected = np.any(val_mask[:, c1_cols], axis=1)

        cluster0_selected = int(c1_selected[:4].sum())
        cluster1_selected = int(c1_selected[4:].sum())

        assert cluster0_selected == 1
        assert cluster1_selected == 2

    def test_categorical_column_map_requires_binary_feature_mask(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
    ):
        """
        categorical_column_map requires binary_feature_mask.
        """
        with pytest.raises(RuntimeError, match="binary_feature_mask required"):
            ClusterDataset(
                data=categorical_validation_base_dataframe,
                cluster_labels=categorical_validation_cluster_labels,
                val_proportion=0.5,
                val_seed=123,
                binary_feature_mask=None,
                categorical_column_map=categorical_validation_column_map,
            )

    def test_categorical_column_map_rejects_unknown_dummy_column_name(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_binary_feature_mask,
    ):
        """
        categorical_column_map should fail if it references a dummy-variable
        column not present in the data.
        """
        bad_map = {
            "C1": ["C1b1", "C1b2"],
            "C2": ["C2b1", "DOES_NOT_EXIST"],
        }

        with pytest.raises(ValueError, match="not found in data"):
            self._build_dataset(
                categorical_validation_base_dataframe=categorical_validation_base_dataframe,
                categorical_validation_cluster_labels=categorical_validation_cluster_labels,
                categorical_validation_column_map=bad_map,
                categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            )

    def test_categorical_column_map_rejects_duplicate_dummy_column_within_category(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_binary_feature_mask,
    ):
        """
        A single original category should not list the same dummy column twice.
        """
        bad_map = {
            "C1": ["C1b1", "C1b1"],
            "C2": ["C2b1", "C2b2"],
        }

        with pytest.raises(ValueError, match="Duplicate dummy columns found"):
            self._build_dataset(
                categorical_validation_base_dataframe=categorical_validation_base_dataframe,
                categorical_validation_cluster_labels=categorical_validation_cluster_labels,
                categorical_validation_column_map=bad_map,
                categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            )

    def test_categorical_column_map_rejects_dummy_column_used_in_multiple_categories(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_binary_feature_mask,
    ):
        """
        A dummy column should belong to only one grouped categorical.
        """
        bad_map = {
            "C1": ["C1b1", "C1b2"],
            "C2": ["C1b2", "C2b2"],
        }

        with pytest.raises(ValueError, match="appear in more than one category"):
            self._build_dataset(
                categorical_validation_base_dataframe=categorical_validation_base_dataframe,
                categorical_validation_cluster_labels=categorical_validation_cluster_labels,
                categorical_validation_column_map=bad_map,
                categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            )

    def test_categorical_column_map_rejects_nonbinary_dummy_column(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        Any dummy column listed in categorical_column_map must be marked True in
        binary_feature_mask.
        """
        bad_binary_mask = categorical_validation_binary_feature_mask.copy()
        bad_binary_mask[4] = False

        with pytest.raises(ValueError, match="is listed in categorical_column_map but is not marked True"):
            self._build_dataset(
                categorical_validation_base_dataframe=categorical_validation_base_dataframe,
                categorical_validation_cluster_labels=categorical_validation_cluster_labels,
                categorical_validation_column_map=categorical_validation_column_map,
                categorical_validation_binary_feature_mask=bad_binary_mask,
            )

    

    def test_repr_includes_validation_units_summary(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        __repr__ should include validation unit information using original
        categorical names.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
        )
        text = repr(ds)

        assert "Validation units" in text
        assert "'C1'" in text
        assert "'C2'" in text
        assert "'C1b1'" in text
        assert "'C1b2'" in text
        assert "'C2b1'" in text
        assert "'C2b2'" in text

    def test_end_to_end_grouped_categorical_behavior(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        Strong integration-style test for grouped categorical behavior.
        """
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
            val_proportion=0.5,
            val_seed=777,
        )
        val_mask = ds.val_mask.cpu().numpy()

        grouped_categories = {
            "C1": ["C1b1", "C1b2"],
            "C2": ["C2b1", "C2b2"],
        }

        for cat_name, dummy_cols in grouped_categories.items():
            dummy_indices = [ds.feature_names.index(col) for col in dummy_cols]

            first_col_pattern = val_mask[:, dummy_indices[0]]
            for idx in dummy_indices[1:]:
                assert np.array_equal(first_col_pattern, val_mask[:, idx])

            assert int(first_col_pattern.sum()) == 2

        for ordinary in ["X1", "X2", "B1", "B2"]:
            assert ordinary in ds.validation_units
            assert ds.validation_units[ordinary]["kind"] in ["binary", "continuous"]

    # ---------------------------------------------------------------------
    # run_cissvae integration tests for categorical_column_map behavior
    # ---------------------------------------------------------------------

    def test_categorical_column_map_masks_grouped_dummy_columns_together(self, minimal_params):
        """
        Test that categorical_column_map causes validation masking to happen at the
        original categorical-variable level rather than independently for each dummy
        column.

        Expected behavior:
        - validation_units should contain original category names (e.g. "C1", "C2")
        - dummy columns belonging to the same category should not appear as separate
        validation units
        - if one dummy column in a grouped category is selected for validation in a
        given row, all dummy columns in that grouped category should be selected
        in that same row
        """
        data = pd.DataFrame(
            {
                "X1":   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "X2":   [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "B1":   [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "B2":   [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "C1b1": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "C1b2": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "C2b1": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                "C2b2": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            }
        )

        binary_feature_mask = np.array(
            [False, False, True, True, True, True, True, True],
            dtype=bool,
        )

        categorical_column_map = {
            "C1": ["C1b1", "C1b2"],
            "C2": ["C2b1", "C2b2"],
        }

        clusters = np.array([0, 0, 0, 1, 1, 1], dtype=int)

        params = minimal_params.copy()
        params.update(
            {
                "return_dataset": True,
                "return_model": False,
                "return_clusters": True,
                "return_silhouettes": False,
                "return_history": False,
                "clusters": clusters,
                "binary_feature_mask": binary_feature_mask,
                "categorical_column_map": categorical_column_map,
                "val_proportion": 0.5,
                "seed": 42,
            }
        )

        result = run_cissvae(data, **params)

        # Expected return order:
        # imputed_dataset, dataset, clusters
        imputed_dataset, dataset, returned_clusters = result

        assert isinstance(imputed_dataset, pd.DataFrame)
        assert isinstance(dataset, ClusterDataset)
        assert isinstance(returned_clusters, np.ndarray)

        assert np.array_equal(returned_clusters, clusters)

        assert "C1" in dataset.validation_units
        assert "C2" in dataset.validation_units

        assert dataset.validation_units["C1"]["kind"] == "categorical"
        assert dataset.validation_units["C2"]["kind"] == "categorical"

        assert "C1b1" not in dataset.validation_units
        assert "C1b2" not in dataset.validation_units
        assert "C2b1" not in dataset.validation_units
        assert "C2b2" not in dataset.validation_units

        c1_cols = [dataset.feature_names[i] for i in dataset.validation_units["C1"]["cols"]]
        c2_cols = [dataset.feature_names[i] for i in dataset.validation_units["C2"]["cols"]]

        assert c1_cols == ["C1b1", "C1b2"]
        assert c2_cols == ["C2b1", "C2b2"]

        val_mask = dataset.val_mask.cpu().numpy()

        c1_mask_1 = val_mask[:, dataset.feature_names.index("C1b1")]
        c1_mask_2 = val_mask[:, dataset.feature_names.index("C1b2")]
        c2_mask_1 = val_mask[:, dataset.feature_names.index("C2b1")]
        c2_mask_2 = val_mask[:, dataset.feature_names.index("C2b2")]

        assert np.array_equal(c1_mask_1, c1_mask_2)
        assert np.array_equal(c2_mask_1, c2_mask_2)

        assert dataset.categorical_column_map == categorical_column_map


    def test_categorical_column_map_excludes_partially_observed_grouped_rows(self, minimal_params):
        """
        Test that a row is not eligible for grouped categorical validation masking
        when one dummy column in that grouped category is missing.
        """
        data = pd.DataFrame(
            {
                "X1":   [1.0, 2.0, 3.0, 4.0],
                "X2":   [10.0, 20.0, 30.0, 40.0],
                "B1":   [0.0, 1.0, 0.0, 1.0],
                "B2":   [1.0, 0.0, 1.0, 0.0],
                "C1b1": [np.nan, 0.0, 1.0, 0.0],
                "C1b2": [0.0,    1.0, 0.0, 1.0],
                "C2b1": [1.0, 1.0, 0.0, 0.0],
                "C2b2": [0.0, 0.0, 1.0, 1.0],
            }
        )

        binary_feature_mask = np.array(
            [False, False, True, True, True, True, True, True],
            dtype=bool,
        )

        categorical_column_map = {
            "C1": ["C1b1", "C1b2"],
            "C2": ["C2b1", "C2b2"],
        }

        clusters = np.array([0, 0, 1, 1], dtype=int)

        params = minimal_params.copy()
        params.update(
            {
                "return_dataset": True,
                "return_model": False,
                "return_clusters": True,
                "return_silhouettes": False,
                "return_history": False,
                "clusters": clusters,
                "binary_feature_mask": binary_feature_mask,
                "categorical_column_map": categorical_column_map,
                "val_proportion": 0.5,
                "seed": 42,
            }
        )

        result = run_cissvae(data, **params)
        imputed_dataset, dataset, returned_clusters = result

        assert isinstance(imputed_dataset, pd.DataFrame)
        assert isinstance(dataset, ClusterDataset)
        assert isinstance(returned_clusters, np.ndarray)

        val_mask = dataset.val_mask.cpu().numpy()
        c1b1_idx = dataset.feature_names.index("C1b1")
        c1b2_idx = dataset.feature_names.index("C1b2")

        # Row 0 has partial missingness for grouped category C1 and should not be
        # eligible for grouped C1 validation masking.
        assert not val_mask[0, c1b1_idx]
        assert not val_mask[0, c1b2_idx]

    ## =========================
    ## Added tests for activation groups
    ## =========================

    def test_activation_groups_structure(
    self,
    categorical_validation_base_dataframe,
    categorical_validation_cluster_labels,
    categorical_validation_column_map,
    categorical_validation_binary_feature_mask,
    ):
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
        )

        ag = ds.activation_groups

        # Required keys
        assert "continuous" in ag
        assert "binary" in ag
        assert "C1" in ag
        assert "C2" in ag

        # Categorical groups must have multiple columns
        assert len(ag["C1"]) == 2
        assert len(ag["C2"]) == 2

    def test_activation_groups_column_indices_correct(
    self,
    categorical_validation_base_dataframe,
    categorical_validation_cluster_labels,
    categorical_validation_column_map,
    categorical_validation_binary_feature_mask,
    ):
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
        )

        ag = ds.activation_groups

        # Continuous
        cont_cols = [ds.feature_names[i] for i in ag["continuous"]]
        assert set(cont_cols) == {"X1", "X2"}

        # Binary (should NOT include categorical dummies)
        bin_cols = [ds.feature_names[i] for i in ag["binary"]]
        assert set(bin_cols) == {"B1", "B2"}

        # Categorical
        c1_cols = [ds.feature_names[i] for i in ag["C1"]]
        c2_cols = [ds.feature_names[i] for i in ag["C2"]]

        assert c1_cols == ["C1b1", "C1b2"]
        assert c2_cols == ["C2b1", "C2b2"]

    def test_activation_groups_no_overlap(
    self,
    categorical_validation_base_dataframe,
    categorical_validation_cluster_labels,
    categorical_validation_column_map,
    categorical_validation_binary_feature_mask,
    ):
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
        )

        ag = ds.activation_groups

        all_indices = []
        for cols in ag.values():
            all_indices.extend(cols)

        assert len(all_indices) == len(set(all_indices))

    def test_activation_groups_covers_all_columns(
    self,
    categorical_validation_base_dataframe,
    categorical_validation_cluster_labels,
    categorical_validation_column_map,
    categorical_validation_binary_feature_mask,
):
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
        )

        ag = ds.activation_groups

        all_indices = set()
        for cols in ag.values():
            all_indices.update(cols)

        expected = set(range(len(ds.feature_names)))

        assert all_indices == expected

    def test_categorical_columns_not_in_binary_group(
    self,
    categorical_validation_base_dataframe,
    categorical_validation_cluster_labels,
    categorical_validation_column_map,
    categorical_validation_binary_feature_mask,
):
        ds = self._build_dataset(
            categorical_validation_base_dataframe=categorical_validation_base_dataframe,
            categorical_validation_cluster_labels=categorical_validation_cluster_labels,
            categorical_validation_column_map=categorical_validation_column_map,
            categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
        )

        ag = ds.activation_groups

        binary_set = set(ag["binary"])
        categorical_set = set(ag["C1"]) | set(ag["C2"])

        assert binary_set.isdisjoint(categorical_set)


class DummyModel(torch.nn.Module):
    def __init__(self, recon):
        super().__init__()
        self.recon = recon

    def forward(self, X, C, deterministic=True):
        return self.recon, None, None

def test_compute_val_metrics_exclude_ignored_columns():
    """
    Ensure ignored columns do NOT contribute to:
    - MSE (continuous)
    - BCE (binary)
    - CE (categorical)
    """

    import numpy as np
    import pandas as pd
    import torch

    # ---------------------------------------
    # Dataset with all three types
    # ---------------------------------------
    df = pd.DataFrame({
        "ignored_cont": [10.0, 20.0, 30.0, 40.0],
        "active_cont":  [1.0,  2.0,  3.0,  4.0],

        "ignored_bin":  [0, 1, 0, 1],
        "active_bin":   [1, 0, 1, 0],

        # categorical (one-hot)
        "cat_a": [1, 0, 1, 0],
        "cat_b": [0, 1, 0, 1],
    })

    clusters = np.zeros(len(df))

    dataset = ClusterDataset(
        data=df,
        cluster_labels=clusters,
        val_proportion=1.0,
        columns_ignore=["ignored_cont", "ignored_bin"],
        binary_feature_mask=[False, False, True, True, True, True],
        categorical_column_map={"cat": ["cat_a", "cat_b"]},
    )

    true = dataset.val_data.clone()
    recon = torch.zeros_like(true)

    # ---------------------------------------
    # Index mapping
    # ---------------------------------------
    idx = {name: dataset.feature_names.index(name) for name in df.columns}

    # ---------------------------------------
    # 1. Continuous (normalized space → already correct)
    # ---------------------------------------
    recon[:, idx["active_cont"]] = true[:, idx["active_cont"]]
    recon[:, idx["ignored_cont"]] = 9999.0  # inject error

    # ---------------------------------------
    # 2. Binary (MUST be logits, not probabilities)
    # ---------------------------------------
    eps = 1e-6

    def to_logit(x):
        x = x.clamp(eps, 1 - eps)
        return torch.log(x / (1 - x))

    # Active binary → perfect reconstruction
    recon[:, idx["active_bin"]] = to_logit(true[:, idx["active_bin"]])

    # Ignored binary → wrong logits
    recon[:, idx["ignored_bin"]] = -10.0  # sigmoid ≈ 0

    # ---------------------------------------
    # 3. Categorical (must be logits)
    # ---------------------------------------
    cat_cols = [idx["cat_a"], idx["cat_b"]]

    # Perfect reconstruction → very confident logits
    recon[:, cat_cols] = true[:, cat_cols] * 10.0  # strong separation

    # ---------------------------------------
    # Model
    # ---------------------------------------
    model = DummyModel(recon)

    # ---------------------------------------
    # Compute metrics
    # ---------------------------------------
    imputation_error, mse, bce, ce = compute_val_mse(
        model, dataset, debug=True
    )

    # ---------------------------------------
    # Assertions
    # ---------------------------------------
    assert mse < 1e-8, "Ignored continuous affected MSE"
    assert bce < 1e-5, "Ignored binary affected BCE"
    assert ce  < 1e-4, "Ignored categorical affected CE"


def test_partial_ignore_categorical_raises():
    df = pd.DataFrame({
        "cat_a": [1, 0, 1],
        "cat_b": [0, 1, 0],
    })

    with pytest.raises(ValueError):
        ClusterDataset(
            data=df,
            cluster_labels=np.zeros(len(df)),
            val_proportion=0.5,
            columns_ignore=["cat_a"],  # partial ignore
            binary_feature_mask=[True, True],
            categorical_column_map={"cat": ["cat_a", "cat_b"]},
        )

from ciss_vae.training.train_refit import impute_and_refit_loop
from ciss_vae.utils.helpers import compute_val_mse


def add_missing(df, n=20, seed=123):
    rng = np.random.default_rng(seed)
    df = df.copy()

    total = df.size
    idx = rng.choice(total, size=min(n, total), replace=False)

    flat = df.values.flatten()
    flat[idx] = np.nan
    df[:] = flat.reshape(df.shape)

    return df


def get_last_non_nan(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return x[-1]

def test_ignored_columns_all_types_behavior_run_cissvae():
    import numpy as np
    import pandas as pd

    from ciss_vae.training.run_cissvae import run_cissvae

    np.random.seed(123)

    # --------------------------------------------------
    # Dataset with ALL types
    # --------------------------------------------------
    n = 80

    df = pd.DataFrame({
        # continuous
        "ignored_cont": np.random.normal(0, 1000, n),
        "active_cont":  np.random.normal(0, 1, n),

        # binary
        "ignored_bin": np.random.binomial(1, 0.1, n),
        "active_bin":  np.random.binomial(1, 0.5, n),

        # categorical (one-hot)
        "cat_a": np.tile([1,0,0], n // 3 + 1)[:n],
        "cat_b": np.tile([0,1,0], n // 3 + 1)[:n],
        "cat_c": np.tile([0,0,1], n // 3 + 1)[:n],
    })

    # inject missingness
    mask = np.random.choice(df.size, size=40, replace=False)
    df.values.flat[mask] = np.nan

    clusters = np.random.randint(0, 2, size=n)

    categorical_map = {"cat": ["cat_a", "cat_b", "cat_c"]}

    binary_mask = [
        False, False,   # continuous
        True, True,     # binary
        True, True, True  # categorical dummy cols
    ]

    # --------------------------------------------------
    # RUN WITH ignored columns
    # --------------------------------------------------
    imputed_ignore, model_ignore, dataset_ignore, history_ignore = run_cissvae(
        data=df,
        clusters=clusters,
        columns_ignore=["ignored_cont", "ignored_bin"],
        binary_feature_mask=binary_mask,
        categorical_column_map=categorical_map,
        epochs=2,
        max_loops=1,
        epochs_per_loop=1,
        return_dataset=True,
        return_history=True,
        verbose=False
    )

    # --------------------------------------------------
    # RUN WITHOUT ignoring
    # --------------------------------------------------
    imputed_full, model_full, dataset_full, history_full = run_cissvae(
        data=df,
        clusters=clusters,
        columns_ignore=None,
        binary_feature_mask=binary_mask,
        categorical_column_map=categorical_map,
        epochs=2,
        max_loops=1,
        epochs_per_loop=1,
        return_dataset=True,
        return_history=True,
        verbose=False
    )

    # --------------------------------------------------
    # TRAIN LOSS (ignore NA rows)
    # --------------------------------------------------
    def get_last_non_nan(x):
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        return x[-1] if len(x) else np.nan

    train_ignore = get_last_non_nan(history_ignore["train_loss"])
    train_full   = get_last_non_nan(history_full["train_loss"])

    train_mse_ignore = get_last_non_nan(history_ignore["train_mse"])
    train_mse_full   = get_last_non_nan(history_full["train_mse"])

    train_bce_ignore = get_last_non_nan(history_ignore["train_bce"])
    train_bce_full   = get_last_non_nan(history_full["train_bce"])

    print("\n===== TRAIN LOSS DEBUG =====")
    print("ignore:", train_ignore)
    print("full:", train_full)
    print("train_mse_ignore:", train_mse_ignore)
    print("train_mse_full", train_mse_full)

    print("train_bce_ignore:", train_bce_ignore)
    print("train_bce_full", train_bce_full)

    assert np.isfinite(train_ignore)
    assert np.isfinite(train_full)

    # ✔ Training loss NOT differ
    assert abs(train_ignore - train_full) < 0.1
    assert abs(train_mse_ignore - train_mse_full) < 10
    assert abs(train_bce_ignore - train_bce_full) < 1

    # --------------------------------------------------
    # IMPUTATION ERROR
    # --------------------------------------------------
    imp_ignore = get_last_non_nan(history_ignore["imputation_error"])
    imp_full   = get_last_non_nan(history_full["imputation_error"])

    mse_ignore = get_last_non_nan(history_ignore["val_mse"])
    mse_full = get_last_non_nan(history_full["val_mse"])

    bce_ignore = get_last_non_nan(history_ignore["val_bce"])
    bce_full = get_last_non_nan(history_full["val_bce"])

    print("\n===== IMPUTATION ERROR DEBUG =====")
    print("ignore:", imp_ignore)
    print("full:", imp_full)

    print("mse_ignore:", mse_ignore)
    print("mse_full:", mse_full)

    print("bce_ignore:", bce_ignore)
    print("bce_full:", bce_full)

    assert np.isfinite(imp_ignore)
    assert np.isfinite(imp_full)

    # ✔ Imputation error SHOULD NOT be identical
    assert abs(imp_ignore - imp_full) > 0.1
    assert abs(mse_ignore - mse_full) > 0.1
    assert abs(bce_ignore - bce_full) > 0.01


def test_ignored_categorical_group_behavior_run_cissvae():
    import numpy as np
    import pandas as pd
    from ciss_vae.training.run_cissvae import run_cissvae

    np.random.seed(123)

    n = 90

    # --------------------------------------------------
    # Build dataset
    # --------------------------------------------------
    df = pd.DataFrame({
        # continuous
        "ignored_cont": np.random.normal(0, 100, n),
        "active_cont":  np.random.normal(0, 1, n),

        # binary
        "ignored_bin": np.random.binomial(1, 0.5, n),
        "active_bin":  np.random.binomial(1, 0.5, n),

        # categorical GROUP 1 (WILL BE IGNORED)
        "cat_ign_a": np.tile([1,0,0], n // 3 + 1)[:n],
        "cat_ign_b": np.tile([0,1,0], n // 3 + 1)[:n],
        "cat_ign_c": np.tile([0,0,1], n // 3 + 1)[:n],

        # categorical GROUP 2 (ACTIVE)
        "cat_act_a": np.tile([1,0], n // 2 + 1)[:n],
        "cat_act_b": np.tile([0,1], n // 2 + 1)[:n],
    })

    # inject missingness
    mask = np.random.choice(df.size, size=60, replace=False)
    df.values.flat[mask] = np.nan

    clusters = np.random.randint(0, 2, size=n)

    # --------------------------------------------------
    # categorical groups
    # --------------------------------------------------
    categorical_map = {
        "ignored_cat": ["cat_ign_a", "cat_ign_b", "cat_ign_c"],
        "active_cat": ["cat_act_a", "cat_act_b"],
    }

    # --------------------------------------------------
    # binary mask
    # (categorical dummy columns must be TRUE)
    # --------------------------------------------------
    binary_mask = [
        False, False,   # continuous
        True, True,     # binary
        True, True, True,   # ignored categorical
        True, True           # active categorical
    ]

    # --------------------------------------------------
    # IMPORTANT: ignore ALL columns of the categorical group
    # --------------------------------------------------
    ignored_cols = [
        "ignored_cont",
        "ignored_bin",
        "cat_ign_a", "cat_ign_b", "cat_ign_c"
    ]

    # --------------------------------------------------
    # RUN WITH ignored categorical group
    # --------------------------------------------------
    imputed_ignore, model_ignore, dataset_ignore, history_ignore = run_cissvae(
        data=df,
        clusters=clusters,
        columns_ignore=ignored_cols,
        binary_feature_mask=binary_mask,
        categorical_column_map=categorical_map,
        epochs=2,
        max_loops=1,
        epochs_per_loop=1,
        return_dataset=True,
        return_history=True,
        verbose=False
    )

    # --------------------------------------------------
    # RUN WITHOUT ignoring
    # --------------------------------------------------
    imputed_full, model_full, dataset_full, history_full = run_cissvae(
        data=df,
        clusters=clusters,
        columns_ignore=None,
        binary_feature_mask=binary_mask,
        categorical_column_map=categorical_map,
        epochs=2,
        max_loops=1,
        epochs_per_loop=1,
        return_dataset=True,
        return_history=True,
        verbose=False
    )

    # --------------------------------------------------
    # helper
    # --------------------------------------------------
    def get_last_non_nan(x):
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        return x[-1] if len(x) else np.nan

    # --------------------------------------------------
    # TRAIN LOSS
    # --------------------------------------------------
    train_ignore = get_last_non_nan(history_ignore["train_loss"])
    train_full   = get_last_non_nan(history_full["train_loss"])

    print("\n===== TRAIN LOSS DEBUG =====")
    print("ignore:", train_ignore)
    print("full:", train_full)

    assert np.isfinite(train_ignore)
    assert np.isfinite(train_full)

    # Must be identical
    assert abs(train_ignore - train_full) < 0.1

    
    train_mse_ignore = get_last_non_nan(history_ignore["train_mse"])
    train_mse_full   = get_last_non_nan(history_full["train_mse"])

    train_bce_ignore = get_last_non_nan(history_ignore["train_bce"])
    train_bce_full   = get_last_non_nan(history_full["train_bce"])

    train_ce_ignore = get_last_non_nan(history_ignore["train_ce"])
    train_ce_full   = get_last_non_nan(history_full["train_ce"])

    print("\n===== TRAIN LOSS DEBUG =====")
    print("train_mse_ignore:", train_mse_ignore)
    print("train_mse_full", train_mse_full)

    print("train_bce_ignore:", train_bce_ignore)
    print("train_bce_full", train_bce_full)
    print("train_ce_ignore:", train_ce_ignore)
    print("train_ce_full", train_ce_full)

    assert np.isfinite(train_ignore)
    assert np.isfinite(train_full)

    # ✔ Training loss NOT differ
    assert abs(train_mse_ignore - train_mse_full) < 10
    assert abs(train_bce_ignore - train_bce_full) < 10
    assert abs(train_ce_ignore - train_ce_full) < 10


    # --------------------------------------------------
    # IMPUTATION ERROR
    # --------------------------------------------------
    imp_ignore = get_last_non_nan(history_ignore["imputation_error"])
    imp_full   = get_last_non_nan(history_full["imputation_error"])

    mse_ignore = get_last_non_nan(history_ignore["val_mse"])
    mse_full = get_last_non_nan(history_full["val_mse"])

    bce_ignore = get_last_non_nan(history_ignore["val_bce"])
    bce_full = get_last_non_nan(history_full["val_bce"])

    ce_ignore = get_last_non_nan(history_ignore["val_ce"])
    ce_full = get_last_non_nan(history_full["val_ce"])

    print("\n===== IMPUTATION ERROR DEBUG =====")
    print("ignore:", imp_ignore)
    print("full:", imp_full)

    print("mse_ignore:", mse_ignore)
    print("mse_full:", mse_full)

    print("bce_ignore:", bce_ignore)
    print("bce_full:", bce_full)

    print("ce_ignore:", ce_ignore)
    print("ce_full:", ce_full)

    assert np.isfinite(imp_ignore)
    assert np.isfinite(imp_full)

    # ✔ Imputation error SHOULD NOT be identical
    assert abs(imp_ignore - imp_full) > 0.1
    assert abs(mse_ignore - mse_full) > 0.1
    assert abs(bce_ignore - bce_full) > 0.1
    assert abs(ce_ignore - ce_full) > 0.1
