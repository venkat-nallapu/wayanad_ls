# Model Validation and Performance Metrics - Wayanad District

## 1. Summary of Quantitative Performance Metrics

| Metric | Value | Sample Size / Reference | Acceptance Benchmark |
| :--- | :---: | :---: | :---: |
| **Success Rate ROC-AUC (Train)** | `0.8527` | 143 pos / 143 neg | > 0.80 |
| **Prediction Rate ROC-AUC (Val)** | `0.8965` | 61 pos / 61 neg | > 0.80 |
| **Cumulative Success Rate AUC** | `0.8493` | 2,365,030 pixels | > 0.80 |
| **Cumulative Prediction Rate AUC** | `0.8937` | 2,365,030 pixels | > 0.80 |
| **Validation Slides in Top 20% Area (%)** | `83.61` | 61 val slides | > 70.0% |
| **Validation Slides in Top 30% Area (%)** | `91.8` | 61 val slides | > 80.0% |
| **2024 Chooralmala Disaster LSI** | `8.0996` | 11.543°N, 76.155°E | Top decile |
| **2024 Chooralmala District Percentile** | `89.94` | 2,365,030 pixels | > 80th %ile |
| **2024 Chooralmala Susceptibility Zone** | `High` | Zone ID 4 | High / Very High |

## 2. Validation Landslide Distribution Across Susceptibility Zones

| Zone ID | Susceptibility Zone | Area (%) | Validation Landslides | Validation Slides (%) | Density Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Very Low** | 28.28% | 0 | 0.00% | **0.00** |
| 2 | **Low** | 26.91% | 4 | 6.56% | **0.24** |
| 3 | **Moderate** | 21.07% | 5 | 8.20% | **0.39** |
| 4 | **High** | 18.37% | 17 | 27.87% | **1.52** |
| 5 | **Very High** | 5.37% | 35 | 57.38% | **10.69** |

## 3. Real-World Out-of-Sample Benchmark: July 30, 2024 Chooralmala Disaster
- **Event:** Catastrophic debris flow initiating at Chooralmala / Vellarimala, Meppadi.
- **Coordinates:** Latitude 11.5430°N, Longitude 76.1550°E (UTM: Easting 625947.97 m, Northing 1276274.33 m).
- **Model Prediction:** LSI = `8.0996` (**89.94th percentile** of all district terrain).
- **Classified Zone:** **High Susceptibility (Zone 4)**.
- **Verification:** **PASSED**. A statistical FR model trained exclusively on pre-2016 landslide events correctly identified this 2024 disaster site as High susceptibility.
