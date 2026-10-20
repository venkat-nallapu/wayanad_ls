# Spatial Block Cross-Validation Results - Wayanad District

Evaluation of spatial autocorrelation and out-of-region model transferability across North and South blocks.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Spatial Split Latitude** | `11.70°N` | Boundary line dividing North and South blocks |
| **Spatial Split UTM Northing** | `1293500 m` | UTM Zone 43N Y-coordinate demarcation |
| **North Block Landslides** | `45` | Landslide initiation points in North block |
| **North Block Non-Landslides** | `119` | Sampled non-landslide points in North block |
| **North Block Area Share** | `54.74%` | 1,294,624 valid 30m pixels |
| **South Block Landslides** | `159` | Landslide initiation points in South block |
| **South Block Non-Landslides** | `85` | Sampled non-landslide points in South block |
| **South Block Area Share** | `45.26%` | 1,070,406 valid 30m pixels |
| **Fold 1 (Train N -> Test S) ROC-AUC** | `0.8523` | Tested on 159 unseen South landslides |
| **Fold 1 Cumulative AUC** | `0.8232` | Area fraction vs landslide fraction |
| **Fold 1 Top 20% South Area Capture** | `72.33%` | Unseen South slides in top 20% susceptible area |
| **Fold 2 (Train S -> Test N) ROC-AUC** | `0.8375` | Tested on 45 unseen North landslides |
| **Fold 2 Cumulative AUC** | `0.8464` | Area fraction vs landslide fraction |
| **Fold 2 Top 20% North Area Capture** | `77.78%` | Unseen North slides in top 20% susceptible area |
| **Mean Spatial Cross-Validation AUC** | `0.8449` | Average out-of-block prediction AUC |
| **Random 70/30 Split Prediction AUC** | `0.8965` | Step 13 baseline with spatial mixing |
| **Spatial Autocorrelation Optimism Gap** | `0.0516` | Performance difference due to spatial proximity |

### Key Scientific Takeaways
1. **Defensible Spatial Transferability**: Both spatial folds achieve high ROC-AUCs (**0.8523** for Fold 1 and **0.8375** for Fold 2), yielding a mean spatial cross-validation AUC of **0.8449**.
2. **Quantified Spatial Autocorrelation**: The random 70/30 split achieves an AUC of **0.8965**, indicating an optimism gap of **0.0516** (5.16%). This confirms that random splitting modestly overestimates predictive power due to spatial clustering of inventory points.
3. **Geomorphic Robustness**: The spatial models transfer effectively despite the topographic contrast between the gentler northern plateau and the rugged southern escarpment.