# Conditioning Factor Multicollinearity and VIF Analysis

Continuous conditioning factors evaluated across 2,365,030 valid 30m grid cells.

| Factor | Description | $R^2$ with other factors | Tolerance ($1 - R^2$) | Variance Inflation Factor (VIF) | Independence Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `slope` | Slope Angle (deg) | 0.0424 | 0.9576 | **1.044** | PASS (VIF < 5.0) |
| `aspect` | Aspect (deg) | 0.0018 | 0.9982 | **1.002** | PASS (VIF < 5.0) |
| `curvature` | Profile Curvature (m^-1) | 0.0037 | 0.9963 | **1.004** | PASS (VIF < 5.0) |
| `dist_roads` | Distance to Roads (m) | 0.1949 | 0.8051 | **1.242** | PASS (VIF < 5.0) |
| `dist_streams` | Distance to Streams (m) | 0.1796 | 0.8204 | **1.219** | PASS (VIF < 5.0) |

- **Maximum pairwise Pearson $|r|$**: **0.4194** (between `dist_roads` and `dist_streams`). Threshold: $|r| < 0.70$.
- **Maximum Variance Inflation Factor**: **1.242** (`dist_roads`). Threshold: $VIF < 5.0$.
- **Conclusion**: All conditioning factors exhibit low pairwise correlations and near-unity VIF values, satisfying independence assumptions for Frequency Ratio summation.