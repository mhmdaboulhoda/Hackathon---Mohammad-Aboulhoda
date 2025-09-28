# Breast Cancer Wisconsin Dataset — Exploratory Data Analysis

*Generated on:* 2025-09-27 20:53 UTC

## Overview

- Rows: **569**
- Columns (including ID & target): **32**
- Feature columns analysed: **30**
- Duplicate IDs detected: **0**
- Duplicate full records: **0**
- Columns with missing values: **0**

### Diagnosis distribution
| Diagnosis | Count | Share |
|-----------|-------|-------|
| Benign (B) | 357 | 62.74% |
| Malignant (M) | 212 | 37.26% |

_No missing values detected across columns._

## Feature summary statistics (overall)
| Feature | Mean | Median | Std Dev | Min | Max |
|---------|------|--------|---------|-----|-----|
| radius_mean | 14.127 | 13.370 | 3.521 | 6.981 | 28.110 |
| texture_mean | 19.290 | 18.840 | 4.297 | 9.710 | 39.280 |
| perimeter_mean | 91.969 | 86.240 | 24.278 | 43.790 | 188.500 |
| area_mean | 654.889 | 551.100 | 351.605 | 143.500 | 2501.000 |
| smoothness_mean | 0.096 | 0.096 | 0.014 | 0.053 | 0.163 |
| compactness_mean | 0.104 | 0.093 | 0.053 | 0.019 | 0.345 |
| concavity_mean | 0.089 | 0.062 | 0.080 | 0.000 | 0.427 |
| concave points_mean | 0.049 | 0.034 | 0.039 | 0.000 | 0.201 |
| symmetry_mean | 0.181 | 0.179 | 0.027 | 0.106 | 0.304 |
| fractal_dimension_mean | 0.063 | 0.062 | 0.007 | 0.050 | 0.097 |
| radius_se | 0.405 | 0.324 | 0.277 | 0.112 | 2.873 |
| texture_se | 1.217 | 1.108 | 0.551 | 0.360 | 4.885 |
| perimeter_se | 2.866 | 2.287 | 2.020 | 0.757 | 21.980 |
| area_se | 40.337 | 24.530 | 45.451 | 6.802 | 542.200 |
| smoothness_se | 0.007 | 0.006 | 0.003 | 0.002 | 0.031 |
| compactness_se | 0.025 | 0.020 | 0.018 | 0.002 | 0.135 |
| concavity_se | 0.032 | 0.026 | 0.030 | 0.000 | 0.396 |
| concave points_se | 0.012 | 0.011 | 0.006 | 0.000 | 0.053 |
| symmetry_se | 0.021 | 0.019 | 0.008 | 0.008 | 0.079 |
| fractal_dimension_se | 0.004 | 0.003 | 0.003 | 0.001 | 0.030 |
| radius_worst | 16.269 | 14.970 | 4.829 | 7.930 | 36.040 |
| texture_worst | 25.677 | 25.410 | 6.141 | 12.020 | 49.540 |
| perimeter_worst | 107.261 | 97.660 | 33.573 | 50.410 | 251.200 |
| area_worst | 880.583 | 686.500 | 568.856 | 185.200 | 4254.000 |
| smoothness_worst | 0.132 | 0.131 | 0.023 | 0.071 | 0.223 |
| compactness_worst | 0.254 | 0.212 | 0.157 | 0.027 | 1.058 |
| concavity_worst | 0.272 | 0.227 | 0.208 | 0.000 | 1.252 |
| concave points_worst | 0.115 | 0.100 | 0.066 | 0.000 | 0.291 |
| symmetry_worst | 0.290 | 0.282 | 0.062 | 0.157 | 0.664 |
| fractal_dimension_worst | 0.084 | 0.080 | 0.018 | 0.055 | 0.207 |

## Feature means by diagnosis
| Feature | Mean (Benign) | Mean (Malignant) | Difference (M - B) |
|---------|----------------|------------------|---------------------|
| radius_mean | 12.147 | 17.463 | 5.316 |
| texture_mean | 17.915 | 21.605 | 3.690 |
| perimeter_mean | 78.075 | 115.365 | 37.290 |
| area_mean | 462.790 | 978.376 | 515.586 |
| smoothness_mean | 0.092 | 0.103 | 0.010 |
| compactness_mean | 0.080 | 0.145 | 0.065 |
| concavity_mean | 0.046 | 0.161 | 0.115 |
| concave points_mean | 0.026 | 0.088 | 0.062 |
| symmetry_mean | 0.174 | 0.193 | 0.019 |
| fractal_dimension_mean | 0.063 | 0.063 | -0.000 |
| radius_se | 0.284 | 0.609 | 0.325 |
| texture_se | 1.220 | 1.211 | -0.009 |
| perimeter_se | 2.000 | 4.324 | 2.324 |
| area_se | 21.135 | 72.672 | 51.537 |
| smoothness_se | 0.007 | 0.007 | -0.000 |
| compactness_se | 0.021 | 0.032 | 0.011 |
| concavity_se | 0.026 | 0.042 | 0.016 |
| concave points_se | 0.010 | 0.015 | 0.005 |
| symmetry_se | 0.021 | 0.020 | -0.000 |
| fractal_dimension_se | 0.004 | 0.004 | 0.000 |
| radius_worst | 13.380 | 21.135 | 7.755 |
| texture_worst | 23.515 | 29.318 | 5.803 |
| perimeter_worst | 87.006 | 141.370 | 54.364 |
| area_worst | 558.899 | 1422.286 | 863.387 |
| smoothness_worst | 0.125 | 0.145 | 0.020 |
| compactness_worst | 0.183 | 0.375 | 0.192 |
| concavity_worst | 0.166 | 0.451 | 0.284 |
| concave points_worst | 0.074 | 0.182 | 0.108 |
| symmetry_worst | 0.270 | 0.323 | 0.053 |
| fractal_dimension_worst | 0.079 | 0.092 | 0.012 |

## Features most correlated with malignancy
| Rank | Feature | Pearson r |
|------|---------|-----------|
| 1 | concave points_worst | 0.794 |
| 2 | perimeter_worst | 0.783 |
| 3 | concave points_mean | 0.777 |
| 4 | radius_worst | 0.776 |
| 5 | perimeter_mean | 0.743 |
| 6 | area_worst | 0.734 |
| 7 | radius_mean | 0.730 |
| 8 | area_mean | 0.709 |
| 9 | concavity_mean | 0.696 |
| 10 | concavity_worst | 0.660 |

## Key observations
- Dataset is imbalanced with malignant cases representing about 37.3% of samples.
- `concave points_worst` shows the strongest linear relationship with malignancy (r ≈ 0.79).
- Mean `area_worst` differs sharply between classes (benign 558.90 vs malignant 1422.29).
- No missing values detected, so minimal cleaning is required before modeling.
