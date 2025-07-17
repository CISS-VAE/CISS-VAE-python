import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from scipy.stats import truncnorm

def rtruncnorm(n, a=0, mean=5, sd=1):
    return truncnorm.rvs(a, np.inf, loc=mean, scale=sd, size=n)

def introduce_missingness(data, p, k, prop_MCAR=1.0):
    """
    Introduce block-wise MCAR missingness into the biomarker columns.

    Parameters:
    - data: pd.DataFrame with biomarker columns named like Y{biomarker}{time}
    - p: number of time points
    - k: number of biomarkers
    - prop_MCAR: proportion of rows in each block to have missing values

    Returns:
    - modified data with missing values
    """
    data = data.copy()
    block_size = len(data) // 4
    biomarker_cols = [f"Y{b+1}{t+1}" for t in range(p) for b in range(k)]

    for block in range(3):  # 3 MCAR blocks
        start = block * block_size
        end = (block + 1) * block_size
        indices = np.random.choice(range(start, end), size=int(prop_MCAR * block_size), replace=False)

        for idx in indices:
            if block == 0:
                col = np.random.choice(biomarker_cols)
                data.loc[idx, col] = np.nan
            else:
                cols = np.random.choice(biomarker_cols, size=3, replace=False)
                data.loc[idx, cols] = np.nan

    return data

def simulate_biomarker_dataset(n=8000, p=5, k=5, n_clusters=4, prop_MCAR=0.25, seed=42):
    """
    Simulates a biomarker dataset with covariates, temporal structure, and clustered missingness.

    Returns:
    - df_complete: fully observed data
    - df_missing: same structure with missingness introduced
    - clusters: array of KMeans cluster labels based on missingness pattern
    - mask_matrix: binary matrix (observed = 1, missing = 0) for biomarker columns
    """
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4 for block assignment.")

    np.random.seed(seed)

    # --- 1. Covariates
    X = pd.DataFrame({
        "Age": np.random.lognormal(mean=np.log(10), sigma=0.2, size=n),
        "ZipCode": np.random.choice(["10001", "20002", "30003"], size=n),
        "Salary": rtruncnorm(n)
    })

    encoder = OneHotEncoder(sparse_output=False)
    zip_df = pd.DataFrame(encoder.fit_transform(X[["ZipCode"]]), columns=encoder.get_feature_names_out())
    X = pd.concat([X.drop(columns=["ZipCode"]), zip_df], axis=1)

    # --- 2. Latent time series Z
    Z = np.zeros((n, p, 2))
    for cl, (start, end) in enumerate([(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000)], start=1):
        b_cl = np.random.normal(0, 1, size=(2, X.shape[1]))
        for i in range(p):
            B_cl = np.diag(np.random.uniform(0.05, 0.2, size=2)) if i > 0 else None
            b_t = np.random.normal(0, 1, size=(2, X.shape[1]))
            for s in range(start, end):
                eps = np.random.normal(size=(2,))
                Z[s, i, :] = (B_cl @ Z[s, i - 1, :] if i > 0 else 0) + b_t @ X.iloc[s].values + eps

    # --- 3. Outcomes Y
    Y = np.zeros((n, p, k))
    betas = [np.random.normal(0, 1, size=7) for _ in range(k)]

    for i in range(p):
        for j in range(k):
            Y[:, i, j] = (
                Z[:, i, 0] * betas[j][0] + Z[:, i, 1] * betas[j][1] +
                X.iloc[:, 0] * betas[j][2] + X.iloc[:, 1] * betas[j][3] +
                X.iloc[:, 2] * betas[j][4] + X.iloc[:, 3] * betas[j][5] +
                X.iloc[:, 4] * betas[j][6] + np.random.normal(size=n)
            )

    Y_df = pd.DataFrame({f"Y{j+1}{i+1}": Y[:, i, j] for j in range(k) for i in range(p)})
    df_complete = pd.concat([X.reset_index(drop=True), Y_df], axis=1)

    # --- 4. Inject MCAR
    df_missing = introduce_missingness(df_complete.copy(), p, k, prop_MCAR)

    # --- 5. Cluster by missingness pattern
    Y_cols = [col for col in df_missing.columns if col.startswith("Y")]
    mask_matrix = df_missing[Y_cols].notna().astype(int)
    #print(mask_matrix)
    prop_missing = 1 - mask_matrix.mean(axis=1).values.reshape(-1, 1)
    clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit(mask_matrix).labels_

    print(f"Cluster labels assigned: {np.unique(clusters)}")

    return df_complete, df_missing, clusters, mask_matrix

