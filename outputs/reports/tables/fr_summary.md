# Frequency Ratio (FR) Analysis Summary - Wayanad District

Computed from 143 training landslides on 30m master grid.

| Factor | Class ID | Class Label | District Pixels | Area % | Training Slides | Slides % | Frequency Ratio (FR) | Susceptibility Indication |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `slope` | 1 | 0-10 deg (Gentle) | 1,303,908 | 55.13% | 18 | 12.59% | **0.228** | Low Susceptibility (FR < 1.0) |
| `slope` | 2 | 10-20 deg (Moderate) | 764,759 | 32.34% | 69 | 48.25% | **1.492** | Moderate Susceptibility (FR > 1.0) |
| `slope` | 3 | 20-30 deg (Moderately steep) | 222,666 | 9.41% | 51 | 35.66% | **3.788** | High Susceptibility (FR > 1.5) |
| `slope` | 4 | 30-40 deg (Steep) | 63,483 | 2.68% | 5 | 3.50% | **1.303** | Moderate Susceptibility (FR > 1.0) |
| `slope` | 5 | >40 deg (Very steep) | 10,214 | 0.43% | 0 | 0.00% | **0.000** | Zero Slides (Stable / Gap) |
| `aspect` | 1 | North (337.5-22.5 deg) | 307,908 | 13.02% | 27 | 18.88% | **1.450** | Moderate Susceptibility (FR > 1.0) |
| `aspect` | 2 | Northeast (22.5-67.5 deg) | 296,912 | 12.55% | 21 | 14.69% | **1.170** | Moderate Susceptibility (FR > 1.0) |
| `aspect` | 3 | East (67.5-112.5 deg) | 276,901 | 11.71% | 17 | 11.89% | **1.015** | Moderate Susceptibility (FR > 1.0) |
| `aspect` | 4 | Southeast (112.5-157.5 deg) | 295,312 | 12.49% | 16 | 11.19% | **0.896** | Low Susceptibility (FR < 1.0) |
| `aspect` | 5 | South (157.5-202.5 deg) | 315,312 | 13.33% | 13 | 9.09% | **0.682** | Low Susceptibility (FR < 1.0) |
| `aspect` | 6 | Southwest (202.5-247.5 deg) | 279,063 | 11.80% | 18 | 12.59% | **1.067** | Moderate Susceptibility (FR > 1.0) |
| `aspect` | 7 | West (247.5-292.5 deg) | 249,242 | 10.54% | 15 | 10.49% | **0.995** | Low Susceptibility (FR < 1.0) |
| `aspect` | 8 | Northwest (292.5-337.5 deg) | 270,692 | 11.45% | 15 | 10.49% | **0.916** | Low Susceptibility (FR < 1.0) |
| `aspect` | 9 | Flat (Slope < 1 deg) | 73,688 | 3.12% | 1 | 0.70% | **0.224** | Low Susceptibility (FR < 1.0) |
| `curvature` | 1 | Concave (<-0.005) | 397,453 | 16.81% | 23 | 16.08% | **0.957** | Low Susceptibility (FR < 1.0) |
| `curvature` | 2 | Flat (-0.005 to +0.005) | 1,572,589 | 66.49% | 64 | 44.76% | **0.673** | Low Susceptibility (FR < 1.0) |
| `curvature` | 3 | Convex (>+0.005) | 394,988 | 16.70% | 56 | 39.16% | **2.345** | High Susceptibility (FR > 1.5) |
| `dist_roads` | 1 | 0-50 m | 570,169 | 24.11% | 84 | 58.74% | **2.437** | High Susceptibility (FR > 1.5) |
| `dist_roads` | 2 | 50-100 m | 392,550 | 16.60% | 18 | 12.59% | **0.758** | Low Susceptibility (FR < 1.0) |
| `dist_roads` | 3 | 100-200 m | 395,784 | 16.73% | 12 | 8.39% | **0.501** | Low Susceptibility (FR < 1.0) |
| `dist_roads` | 4 | 200-300 m | 178,957 | 7.57% | 10 | 6.99% | **0.924** | Low Susceptibility (FR < 1.0) |
| `dist_roads` | 5 | >300 m | 827,570 | 34.99% | 19 | 13.29% | **0.380** | Low Susceptibility (FR < 1.0) |
| `dist_streams` | 1 | 0-50 m | 319,287 | 13.50% | 33 | 23.08% | **1.709** | High Susceptibility (FR > 1.5) |
| `dist_streams` | 2 | 50-100 m | 268,944 | 11.37% | 20 | 13.99% | **1.230** | Moderate Susceptibility (FR > 1.0) |
| `dist_streams` | 3 | 100-200 m | 414,778 | 17.54% | 24 | 16.78% | **0.957** | Low Susceptibility (FR < 1.0) |
| `dist_streams` | 4 | 200-300 m | 353,053 | 14.93% | 9 | 6.29% | **0.422** | Low Susceptibility (FR < 1.0) |
| `dist_streams` | 5 | >300 m | 1,008,968 | 42.66% | 57 | 39.86% | **0.934** | Low Susceptibility (FR < 1.0) |
| `landcover` | 1 | Tree cover | 2,128,736 | 90.01% | 129 | 90.21% | **1.002** | Moderate Susceptibility (FR > 1.0) |
| `landcover` | 3 | Grassland | 132,718 | 5.61% | 9 | 6.29% | **1.122** | Moderate Susceptibility (FR > 1.0) |
| `landcover` | 4 | Cropland | 71,037 | 3.00% | 0 | 0.00% | **0.000** | Zero Slides (Stable / Gap) |
| `landcover` | 5 | Built-up | 18,024 | 0.76% | 5 | 3.50% | **4.588** | High Susceptibility (FR > 1.5) |
| `landcover` | 6 | Bare / sparse vegetation | 241 | 0.01% | 0 | 0.00% | **0.000** | Zero Slides (Stable / Gap) |
| `landcover` | 7 | Permanent water bodies | 14,181 | 0.60% | 0 | 0.00% | **0.000** | Zero Slides (Stable / Gap) |
| `landcover` | 8 | Herbaceous wetland | 93 | 0.00% | 0 | 0.00% | **0.000** | Zero Slides (Stable / Gap) |