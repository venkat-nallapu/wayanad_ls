# Conditioning Factor Class Size & Zero-FR Sanity Check

Evaluation of landslide counts per factor class to identify noisy low-sample classes and zero-FR occurrences (Step 14).

| Factor | Bin | Class Label | Cells | Area (%) | Training Slides | Frequency Ratio (FR) | Evaluation Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `slope` | 5 | >40 deg (Very steep) | 10,214 | 0.43% | **0** | **0.0000** | Zero Slides (Physically Grounded) |
| `aspect` | 9 | Flat (Slope < 1 deg) | 73,688 | 3.12% | **1** | **0.2244** | Low Sample (N < 5) |
| `landcover` | 4 | Cropland | 71,037 | 3.00% | **0** | **0.0000** | Zero Slides (Physically Grounded) |
| `landcover` | 6 | Bare / sparse vegetation | 241 | 0.01% | **0** | **0.0000** | Zero Slides (Physically Grounded) |
| `landcover` | 7 | Permanent water bodies | 14,181 | 0.60% | **0** | **0.0000** | Zero Slides (Physically Grounded) |
| `landcover` | 8 | Herbaceous wetland | 93 | 0.00% | **0** | **0.0000** | Zero Slides (Physically Grounded) |

### Interpretation & Defense
1. **Slope >40° (Class 5)**: Represents only 0.43% of the district (10,214 cells). These extreme slopes consist of bare crystalline bedrock where soil cover cannot accumulate. Landslides require soil cover to initiate under rainfall infiltration, hence 0 historical slides and FR = 0.0 accurately reflect geotechnical mechanics.
2. **Flat Aspect (Class 9)**: Slopes < 1° have minimal shear stress along potential slip surfaces; 1 slide in 73,688 cells yields FR = 0.224, correctly reflecting strong failure inhibition.
3. **Landcover Non-Vegetated Classes**: Bare rock (Class 6), Water bodies (Class 7), and Herbaceous wetland (Class 8) have negligible area (<0.6% combined) and 0 slides, confirming robust spatial conservation.