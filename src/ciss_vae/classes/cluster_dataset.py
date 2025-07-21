from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import copy

class ClusterDataset(Dataset):
    def __init__(self, data, cluster_labels, val_percent = 0.1, replacement_value = 0, columns_ignore = None):
        """
        Dataset that handles cluster-wise masking and normalization for VAE training.

        Parameters:
            - data (pd.DataFrame, np.ndarray, or torch.Tensor): Input matrix with potential missing values.
            - cluster_labels (array-like): Cluster assignment per sample.
            - val_percent (float): Fraction of non-missing data per cluster to mask for validation.
            - replacement_value (float): Value to fill in missing entries after masking (e.g., 0.0).
            - columns_ignore (list): Optional list of column names (if data is a DataFrame) or indices (if array)
                                    to exclude from validation masking.
        """

        ## set columns ignore 
        if columns_ignore is None:
            self.columns_ignore = []
        else:
            # If columns_ignore is a pandas Index or Series, convert to list
            if hasattr(columns_ignore, "tolist"):
                self.columns_ignore = columns_ignore.tolist()
            else:
                self.columns_ignore = list(columns_ignore)


        # ----------------------------------------
        # Convert input data to numpy
        # ----------------------------------------
        if hasattr(data, 'iloc'):  # pandas DataFrame
            self.indices = torch.tensor(data.index.values, dtype=torch.long)
            self.feature_names = list(data.columns)
            raw_data_np = data.values.astype(np.float32)
            ignore_indices = [i for i, col in enumerate(data.columns) if col in self.columns_ignore]
        elif isinstance(data, np.ndarray):
            self.indices = torch.arange(data.shape[0])
            self.feature_names = [f"V{i+1}" for i in range(data.shape[1])]
            raw_data_np = data.astype(np.float32)
            ignore_indices = self.columns_ignore if isinstance(self.columns_ignore, list) else []
        elif isinstance(data, torch.Tensor):
            self.indices = torch.arange(data.shape[0])
            self.feature_names = [f"V{i+1}" for i in range(data.shape[1])]
            raw_data_np = data.cpu().numpy().astype(np.float32)
            ignore_indices = self.columns_ignore if isinstance(self.columns_ignore, list) else []
        else:
            raise TypeError("Unsupported data format. Must be DataFrame, ndarray, or Tensor.")

        self.raw_data = torch.tensor(raw_data_np, dtype=torch.float32)

        

        # ----------------------------------------
        # Cluster labels to numpy
        # ----------------------------------------
        if hasattr(cluster_labels, 'iloc'):
            cluster_labels_np = cluster_labels.values
        elif isinstance(cluster_labels, np.ndarray):
            cluster_labels_np = cluster_labels
        elif isinstance(cluster_labels, torch.Tensor):
            cluster_labels_np = cluster_labels.cpu().numpy()
        else:
            raise TypeError("Unsupported cluster_labels format. Must be Series, ndarray, or Tensor.")

        self.cluster_labels = torch.tensor(cluster_labels_np, dtype=torch.long)

        self.n_clusters = len(np.unique(cluster_labels_np))

        # ----------------------------------------
        # Validation mask per cluster
        # ----------------------------------------
        val_mask_np = np.zeros_like(raw_data_np, dtype=bool)

        ## for each cluster
        val_mask_np = np.zeros_like(raw_data_np, dtype=bool)
        for cluster_id in np.unique(cluster_labels_np):
            row_idxs = np.where(cluster_labels_np == cluster_id)[0]
            cluster_data = raw_data_np[row_idxs]

            for col in range(cluster_data.shape[1]):
                if col in ignore_indices:
                    continue
                non_missing = [i for i in range(len(row_idxs)) if not np.isnan(cluster_data[i, col])]
                n_val = int(np.floor(len(non_missing) * val_percent))
                if n_val > 0:
                    selected = np.random.choice(non_missing, size=n_val, replace=False)
                    val_mask_np[row_idxs[selected], col] = True

        val_mask_tensor = torch.tensor(val_mask_np, dtype=torch.bool)

        # ----------------------------------------
        # Set aside val_data
        # ----------------------------------------
        self.val_data = self.raw_data.clone()
        self.val_data[~val_mask_tensor] = torch.nan  # keep only validation-masked values

        # ----------------------------------------
        # Combine true + validation-masked missingness
        # ----------------------------------------
        self.data = self.raw_data.clone()
        self.data[val_mask_tensor] = torch.nan  # mask validation entries

        # ----------------------------------------
        # Normalize non-missing entries
        # ----------------------------------------
        ## Compute mean and std on observed (non-NaN) entries
        data_np = self.data.numpy()
        self.feature_means = np.nanmean(data_np, axis=0)
        self.feature_stds = np.nanstd(data_np, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0  # avoid division by zero

        ## Normalize (in-place)
        norm_data_np = (data_np - self.feature_means) / self.feature_stds
        self.data = torch.tensor(norm_data_np, dtype=torch.float32)

        # ----------------------------------------
        # Track missing & replace with value
        # ----------------------------------------
        self.masks = ~torch.isnan(self.data) ## true where value not na
        self.data = torch.where(self.masks, self.data, torch.tensor(replacement_value, dtype=torch.float32))
        self.shape = self.data.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.data[index],            # input with missing replaced
            self.cluster_labels[index], # cluster label
            self.masks[index],          # binary mask
            self.indices[index]         # original row index
        )

    def __repr__(self):
        n, p = self.data.shape
        total_values = n * (p-len(self.columns_ignore))

        ## Percent originally missing (before validation mask)
        original_missing = torch.isnan(self.raw_data).sum().item()
        original_missing_pct = 100 * original_missing / total_values

        ## Percent used for validation (out of non-missing entries)
        val_entries = torch.sum(~torch.isnan(self.val_data)).item()  # number of validation-held entries
        val_pct_of_nonmissing = 100 * val_entries / (total_values - original_missing)

        return (
            f"ClusterDataset(n_samples={n}, n_features={p}, n_clusters={len(torch.unique(self.cluster_labels))})\n"
            f"  • Original missing: {original_missing} / {total_values} "
            f"({original_missing_pct:.2f}%)\n"
            f"  • Validation held-out: {val_entries} "
            f"({val_pct_of_nonmissing:.2f}% of non-missing)\n"
            f"  • .data shape:     {tuple(self.data.shape)}\n"
            f"  • .masks shape:    {tuple(self.masks.shape)}\n"
            f"  • .val_data shape: {tuple(self.val_data.shape)}"
        )

    # ----------------------------------------
    # Added copy method
    # ----------------------------------------
    def copy(self):
        return copy.deepcopy(self)


    def __str__(self):
        return self.__repr__()
