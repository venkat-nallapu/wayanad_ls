# Landslide Susceptibility Zone Statistics - Wayanad District

Classification Method: Fisher-Jenks Natural Breaks (5 classes)

| Zone ID | Susceptibility Zone | LSI Range | Area (km²) | Area (%) | Train Slides | Train Slide % | Val Slides | Val Slide % | Total Slides | Slide Density (slides/km²) | Density Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Very Low** | < 4.86 | 601.94 | 28.28% | 3 | 2.10% | 0 | 0.00% | 3 | **0.0050** | 0.05 |
| 2 | **Low** | 4.86 - 6.00 | 572.87 | 26.91% | 13 | 9.09% | 4 | 6.56% | 17 | **0.0297** | 0.31 |
| 3 | **Moderate** | 6.00 - 7.15 | 448.42 | 21.07% | 13 | 9.09% | 5 | 8.20% | 18 | **0.0401** | 0.42 |
| 4 | **High** | 7.15 - 8.74 | 391.06 | 18.37% | 53 | 37.06% | 17 | 27.87% | 70 | **0.1790** | 1.87 |
| 5 | **Very High** | >= 8.74 | 114.24 | 5.37% | 61 | 42.66% | 35 | 57.38% | 96 | **0.8403** | 8.77 |

**Summary & Physical Verification:**
- **Total District Area:** 2128.53 km² (2,365,030 pixels)
- **Landslide Inventory:** 204 total slides (143 train, 61 val)
- **Monotonic Density:** `CONFIRMED` — Landslide density strictly increases from 0.0050 slides/km² (Very Low) to 0.8403 slides/km² (Very High).
- **High & Very High Concentration:** The High and Very High zones encompass 23.74% of the district area but capture 81.37% of all landslides (85.25% of unseen validation landslides).
