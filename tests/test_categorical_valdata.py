import pytest
import numpy as np
import pandas as pd
import torch
from ciss_vae.classes.cluster_dataset import ClusterDataset


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
            assert ds.validation_units[feature_name]["kind"] == "column"
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

    def test_categorical_column_map_rejects_overlap_with_columns_ignore(
        self,
        categorical_validation_base_dataframe,
        categorical_validation_cluster_labels,
        categorical_validation_column_map,
        categorical_validation_binary_feature_mask,
    ):
        """
        Grouped dummy columns should not also appear in columns_ignore.
        """
        with pytest.raises(ValueError, match="appears in both categorical_column_map and columns_ignore"):
            self._build_dataset(
                categorical_validation_base_dataframe=categorical_validation_base_dataframe,
                categorical_validation_cluster_labels=categorical_validation_cluster_labels,
                categorical_validation_column_map=categorical_validation_column_map,
                categorical_validation_binary_feature_mask=categorical_validation_binary_feature_mask,
                columns_ignore=["C1b1"],
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
            assert ds.validation_units[ordinary]["kind"] == "column"