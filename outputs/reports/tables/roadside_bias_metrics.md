# Roadside Reporting and Observation Bias Analysis

Comparison of historical landslide inventory events against district land area across road proximity tiers.

| Tier | Distance Range | Description | District Area (%) | Train Slides (%) | Val Slides (%) | Total Slides (%) | Training FR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **0–50 m** | Immediate roadside corridor | 24.11% | 58.74% (84) | 73.77% (45) | **63.24%** (129) | **2.437** |
| 2 | **50–100 m** | Proximal buffer | 16.60% | 12.59% (18) | 14.75% (9) | **13.24%** (27) | **0.758** |
| 3 | **100–200 m** | Intermediate buffer | 16.73% | 8.39% (12) | 3.28% (2) | **6.86%** (14) | **0.501** |
| 4 | **200–300 m** | Distal buffer | 7.57% | 6.99% (10) | 3.28% (2) | **5.88%** (12) | **0.924** |
| 5 | **>300 m** | Remote interior / dense forest | 34.99% | 13.29% (19) | 4.92% (3) | **10.78%** (22) | **0.380** |

### Continuous Road Distance Distribution
- **District Land Cells**: Median = **150.0 m** (Mean = 657.2 m)
- **Historical Landslides**: Median = **30.0 m** (Mean = 128.3 m)
- **Median Ratio**: Historical landslides initiate **5.0x closer** to roads than random district locations.

### Geotechnical Interpretation of Observation Bias
1. **Disproportionate Concentration**: Over **58.74%** of training landslides (and **63.24%** of all historical inventory points) fall within **50 meters** of a road, despite this buffer accounting for only **24.11%** of Wayanad's land area ($FR = 2.437$).
2. **Dual Physical and Anthropogenic Drivers**:
   - *Anthropogenic / Physical*: Road construction cuts steep slope toes, alters natural drainage channels, and removes lateral confining stress.
   - *Reporting / Accessibility Bias*: Failures adjacent to roads block transport corridors and are immediately reported by public works agencies, whereas natural slope failures in remote, uninhabited Western Ghats forest tracts frequently remain unrecorded in historical databases.