# Using `create_missingness_prop_matrix`: A Complete Guide

The `create_missingness_prop_matrix()` function creates a matrix showing the proportion of missing values for each sample–feature combination across multiple timepoints. This matrix can then be used for **sample clustering** with `cluster_on_missing_prop()` or as input to the complete CISS-VAE pipeline via `run_cissvae()`.

## Overview

The function returns a **n_samples × n_features** matrix where each cell contains a value between 0 and 1 representing:

```
proportion = (number of missing timepoints) / (total timepoints)
```

- **0.0** = no missing values for that sample-feature combination
- **1.0** = all timepoints missing for that sample-feature combination

---

## Wide Format Examples

### Example 1: Basic Wide Format with Regex Parsing

```python
import pandas as pd
import numpy as np
from ciss_vae.utils.matrix import create_missingness_prop_matrix

# Create sample wide-format data
# Columns follow pattern: feature_timepoint
data_wide = pd.DataFrame({
    'sample_id': ['mouse1', 'mouse2', 'mouse3'],
    'glucose_t0': [100, 95, np.nan],
    'glucose_t1': [110, np.nan, 105],
    'glucose_t2': [np.nan, 100, 115],
    'insulin_t0': [5.2, 4.8, 5.0],
    'insulin_t1': [np.nan, np.nan, 5.5],
    'insulin_t2': [6.0, 5.1, np.nan]
})

# Basic usage - regex automatically extracts features
prop_matrix = create_missingness_prop_matrix(
    data_wide,
    format="wide",
    sample_col="sample_id"
)

print(prop_matrix)
```

**Output:**
```
         glucose   insulin
mouse1  0.333333  0.333333
mouse2  0.333333  0.333333  
mouse3  0.333333  0.333333
```

### Example 2: Custom Regex Pattern

```python
# Data with different naming convention: feature.timepoint
data_custom = pd.DataFrame({
    'ID': ['A', 'B', 'C'],
    'weight.day1': [25.0, np.nan, 27.0],
    'weight.day7': [24.5, 26.0, np.nan],
    'height.day1': [10.0, 9.5, np.nan],
    'height.day7': [10.2, np.nan, 9.8]
})

prop_matrix = create_missingness_prop_matrix(
    data_custom,
    format="wide",
    sample_col="ID",
    wide_regex=r"^(?P<feature>.+?)\.(?P<time>.+)$"  # Custom pattern
)
```

**Output:**

```
    weight  height
ID                
A      0.0     0.0
B      0.5     0.5
C      0.5     0.5
```


### Example 3: Explicit Column Mapping

```python
# When column names don't follow a pattern
data_mixed = pd.DataFrame({
    'subject': ['S1', 'S2'],
    'baseline_glucose': [90, np.nan],
    'followup_glucose': [95, 88],
    'glucose_final': [np.nan, 92],
    'insulin_start': [4.0, 3.8],
    'insulin_end': [4.5, np.nan]
})

# Explicitly map columns to features
column_mapping = {
    'baseline_glucose': 'glucose',
    'followup_glucose': 'glucose', 
    'glucose_final': 'glucose',
    'insulin_start': 'insulin',
    'insulin_end': 'insulin'
}

prop_matrix = create_missingness_prop_matrix(
    data_mixed,
    format="wide",
    sample_col="subject",
    wide_col_to_feature=column_mapping
)

print(prop_matrix)
```

**Output**

```
          glucose  insulin
subject                   
S1       0.333333      0.0
S2       0.333333      0.5
```

### Example 4: MultiIndex Columns

```python
# Create MultiIndex columns
arrays = [
    ['glucose', 'glucose', 'insulin', 'insulin'],
    ['t0', 't1', 't0', 't1']
]
columns = pd.MultiIndex.from_arrays(arrays, names=['feature', 'time'])

data_multi = pd.DataFrame({
    ('glucose', 't0'): [100, np.nan],
    ('glucose', 't1'): [110, 105],
    ('insulin', 't0'): [5.0, 4.8],
    ('insulin', 't1'): [np.nan, 5.2]
}, index=['mouse1', 'mouse2'])

data_multi.columns = columns

prop_matrix = create_missingness_prop_matrix(
    data_multi,
    format="wide",
    wide_multiindex_feature_level="feature"  # Use 'feature' level
)

print(prop_matrix)
```

**Output**

```
        glucose  insulin
mouse1      0.0      0.5
mouse2      0.5      0.0
```

---

## Long Format Examples

### Example 5: Basic Long Format

```python
# Create long-format data
data_long = pd.DataFrame({
    'subject_id': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
    'biomarker': ['glucose', 'glucose', 'insulin', 'insulin'] * 2,
    'timepoint': ['t0', 't1', 't0', 't1'] * 2,
    'measurement': [100, 110, 5.0, np.nan, np.nan, 105, 4.8, 5.2]
})

prop_matrix = create_missingness_prop_matrix(
    data_long,
    format="long",
    sample_col="subject_id",
    feature_col="biomarker", 
    time_col="timepoint",
    value_col="measurement"
)

print(prop_matrix)
```

**Output:**
```
biomarker  glucose  insulin
subject_id                 
A              0.0      0.5
B              0.5      0.0
```

### Example 6: Long Format with Expected Timepoints

```python
# Some timepoints might be completely missing from the data
incomplete_long = pd.DataFrame({
    'mouse': ['M1', 'M1', 'M2', 'M2'],
    'feature': ['weight', 'weight', 'weight', 'height'],
    'day': ['d1', 'd3', 'd1', 'd1'],  # Missing d2, d3 for some combinations
    'value': [25.0, 27.0, np.nan, 10.0]
})

# Specify the complete timepoint grid
expected_days = ['d1', 'd2', 'd3']

prop_matrix = create_missingness_prop_matrix(
    incomplete_long,
    format="long",
    sample_col="mouse",
    feature_col="feature",
    time_col="day", 
    value_col="value",
    expected_timepoints=expected_days  # Absent rows counted as missing
)

print(prop_matrix)
```

**Output:**
```
feature  height    weight
mouse                    
M1          1.0  0.333333  # height missing at d2,d3; weight missing at d2
M2          1.0  0.666667  # height missing at d2,d3; weight missing at d1,d2
```

---

## Integration with Sample Clustering

The primary use case is to cluster **samples** based on their missingness patterns using `cluster_on_missing_prop`:

```python
from ciss_vae.utils.clustering import cluster_on_missing_prop

data_wide = pd.DataFrame({
    'sample_id':  ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5'],
    'glucose_t0': [100,      95,       np.nan,   102,      98     ],
    'glucose_t1': [110,      np.nan,   105,      108,      np.nan ],
    'glucose_t2': [115,      100,      107,   118,   112    ],
    'insulin_t0': [5.2,      4.8,      np.nan,      np.nan,   5.1    ],
    'insulin_t1': [np.nan,   np.nan,   5.5,      5.3,      5.4    ],
    'insulin_t2': [6.0,      5.1,      np.nan,   np.nan,      np.nan ],
})

print(data_wide)

# Create missingness proportion matrix
prop_matrix = create_missingness_prop_matrix(
    data_wide, 
    format="wide", 
    sample_col="sample_id"
)

print(prop_matrix)

# Cluster samples based on similar missingness patterns across features
clusters, silhouette = cluster_on_missing_prop(
    prop_matrix,
    n_clusters=2,
    metric="cosine",
    scale_features=True
)

print("Clusters:", clusters)
print("Silhouette score:", silhouette)

```


---

## Integration with CISS-VAE Pipeline

The proportion matrix can be used directly in the complete CISS-VAE pipeline via the `missingness_proportion_matrix` parameter:

```python
from ciss_vae.training.run_cissvae import run_cissvae

prop_matrix = create_missingness_prop_matrix(
    data_wide,
    format="wide", 
    sample_col="sample_id"
)

imputed_data, model = run_cissvae(
    data=data_wide.drop('sample_id', axis=1),
    missingness_proportion_matrix=prop_matrix,  # Use custom matrix
    n_clusters=2,
    scale_features=True,  # Scale the proportion matrix features
    verbose=True
)
```

---

## Complete Workflow Example

Here's a complete example showing the entire pipeline:

```python
import pandas as pd
import numpy as np
from ciss_vae.utils.matrix import create_missingness_prop_matrix
from ciss_vae.utils.clustering import cluster_on_missing_prop
from ciss_vae.training.run_cissvae import run_cissvae

# 1. Create longitudinal data with complex missingness patterns
np.random.seed(42)
data = pd.DataFrame({
    'patient_id': [f'P{i:03d}' for i in range(1, 101)],
    # Glucose measurements - some patients miss early timepoints
    'glucose_baseline': np.random.normal(95, 10, 100),
    'glucose_month1': np.where(np.random.random(100) < 0.3, np.nan, np.random.normal(92, 12, 100)),
    'glucose_month3': np.where(np.random.random(100) < 0.2, np.nan, np.random.normal(90, 11, 100)),
    'glucose_month6': np.where(np.random.random(100) < 0.1, np.nan, np.random.normal(88, 10, 100)),
    # HbA1c - different missingness pattern
    'hba1c_baseline': np.where(np.random.random(100) < 0.1, np.nan, np.random.normal(6.5, 0.8, 100)),
    'hba1c_month1': np.where(np.random.random(100) < 0.4, np.nan, np.random.normal(6.3, 0.9, 100)),
    'hba1c_month3': np.where(np.random.random(100) < 0.3, np.nan, np.random.normal(6.1, 0.8, 100)),
    'hba1c_month6': np.where(np.random.random(100) < 0.2, np.nan, np.random.normal(5.9, 0.7, 100)),
})

# 2. Create missingness proportion matrix
prop_matrix = create_missingness_prop_matrix(
    data,
    format="wide",
    sample_col="patient_id",
    wide_regex=r"^(?P<feature>.+?)_(?P<time>.+)$"
)

print("Missingness Proportion Matrix:")
print(prop_matrix.head())

# 3. Cluster by missingness patterns
clusters, sil_score = cluster_on_missing_prop(
    prop_matrix,
    n_clusters=2,
    metric="cosine",
    scale_features=True,
    seed=42
)

# 4. Run complete CISS-VAE pipeline
main_data = data.drop('patient_id', axis=1)

imputed_data, model, history = run_cissvae(
    data=main_data,
    missingness_proportion_matrix=prop_matrix,
    n_clusters=2,
    scale_features=True,
    epochs=100,
    verbose=True,
    return_history=True
)

print(f"\nOriginal data shape: {main_data.shape}")
print(f"Imputed data shape: {imputed_data.shape}")
print(f"Missing values - Original: {main_data.isnull().sum().sum()}")
print(f"Missing values - Imputed: {imputed_data.isnull().sum().sum()}")
```

---

## Common Use Cases

### 1. **Longitudinal Clinical Data**
```python
# Patient biomarkers measured at multiple visits
clinical_data = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003'],
    'glucose_baseline': [95, np.nan, 102],
    'glucose_month3': [98, 90, np.nan], 
    'glucose_month6': [np.nan, 88, 105],
    'hba1c_baseline': [6.2, 6.8, np.nan],
    'hba1c_month3': [6.0, np.nan, 6.5],
    'hba1c_month6': [5.8, 6.3, 6.1]
})

prop_matrix = create_missingness_prop_matrix(
    clinical_data,
    format="wide",
    sample_col="patient_id"
)

# Use in CISS-VAE pipeline
imputed_clinical, model = run_cissvae(
    data=clinical_data.drop('patient_id', axis=1),
    missingness_proportion_matrix=prop_matrix,
    n_clusters=2
)
```

### 2. **Multi-omics Time Series**
```python
# Gene expression across timepoints
omics_long = pd.DataFrame({
    'sample': ['S1', 'S1', 'S2', 'S2'] * 3,
    'gene': ['GENE1', 'GENE2', 'GENE1', 'GENE2'] * 3,
    'timepoint': ['0h', '0h', '0h', '0h', '6h', '6h', '6h', '6h', '12h', '12h', '12h', '12h'],
    'expression': [100, 200, np.nan, 180, 110, np.nan, 95, 190, 105, 210, 90, np.nan]
})

prop_matrix = create_missingness_prop_matrix(
    omics_long,
    format="long",
    sample_col="sample",
    feature_col="gene",
    time_col="timepoint", 
    value_col="expression"
)

# Convert long data to wide for CISS-VAE
omics_wide = omics_long.pivot_table(
    index='sample', 
    columns=['gene', 'timepoint'], 
    values='expression'
)

imputed_omics, model = run_cissvae(
    data=omics_wide,
    missingness_proportion_matrix=prop_matrix,
    n_clusters=2
)
```

---

## Tips and Best Practices

1. **Sample Clustering**: `cluster_on_missing_prop` clusters **samples (rows)** based on their missingness profiles across features. This groups subjects with similar patterns of missing data.

2. **Data format**: Provide the input as a matrix or DataFrame with **rows = samples** and **columns = features**, where entries are missingness proportions in `[0, 1]`.

3. **Regex helpers (optional)**: If you need to parse wide column names like `feature_timepoint`, adjust your regex (default `r"^(?P<feature>.+?)_(?P<time>[^_]+)$"`) to match your naming convention before constructing the matrix.

4. **Expected timepoints**: When starting from long-format data, always specify the complete set of `expected_timepoints` before pivoting to wide format. Otherwise, unobserved timepoints won’t be registered as missing.

5. **Feature filtering**: Apply a threshold like `min_timepoints_per_feature` during preprocessing to exclude features that don’t have sufficient data across samples.

6. **Scaling**: Use `scale_features=True` if features differ greatly in their overall missingness rates. This standardizes the columns so no single feature dominates the clustering.

7. **Metric selection**: For proportion vectors, `metric="cosine"` often highlights **pattern similarity** in missingness more effectively than `"euclidean"`.



