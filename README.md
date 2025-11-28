# Reef Island Shoreline Modeling with Machine Learning

## Repository Structure
- **codes/**: Source code for data processing, modeling, and evaluation (see `codes/READM.md` for usage instructions).  
- **deliverables/**: Project documentation and final outputs.  
- **logbook/**: Research logbook.

## Project Overview
Reef islands in the Maldives are low-lying accumulations of biogenic carbonate sediment whose shorelines dynamically respond to interacting **oceanographic, climatic, and geomorphic drivers**. Anticipating shoreline change is critical for risk-informed coastal planning and climate adaptation.  

This project systematically develops **nine machine learning approaches** (including Random Forest, LightGBM, XGBoost, Stacking, Ridge Regression, KNN, and neural networks) to predict **along-transect shoreline positions**. A **hybrid transect framework** method is introduced to minimize orientation bias and reduce satellite imagery noise.  

The study integrates **multi-source data resorces**, including:  
- **Satellite remote sensing** (Sentinel-2, Landsat 8/9, NDVI vegetation index)  
- **Topography** (NASADEM-derived coastal slopes)  
- **Wind & wave climate** (ERA5 reanalysis data)  
- **Bathymetry** (reef width, reef slope, reef flat depth from improved gridded bathymetric data)  
- **Monsoon** (seasonal forcing regimes in the Maldives)  

## Key Findings
- **Single-island experiments**: Stacking (LightGBM + XGBoost + Ridge) achieved the highest accuracy (mean R² = 0.96), closely followed by Random Forest and XGBoost.  
- **Multi-island experiments**: Random Forest performed best (R² ≈ 0.95), with Stacking and tuned neural networks also exceeding R² > 0.90.  
- **Future prediction (2016–2023 → 2024)**: Random Forest achieved R² ≈ 0.94, demonstrating strong ability to predict near-future shoreline states.  
- **Cross-island generalization**: All models showed poor transferability to unseen islands (R² < 0.2), highlighting the need for larger and more diverse datasets.  
- **Feature importance**: Reef geomorphology and local terrain slope are the dominant predictors, reflecting strong geomorphic control over shoreline dynamics.  

**Performance vs. Efficiency:**  
- Random Forest consistently produced the most accurate predictions but required **>384× more memory** (≈470 MB) than other models (≤5 MB).  
- Stacking models provided high accuracy with significantly lower memory usage.  

## Contributions
- First systematic application of **machine learning frameworks** for multi-factor reef island shape prediction in the Maldives.  
- Developed a **hybrid transect design** for improved shoreline extraction and orientation handling.  
- Demonstrated the potential of **ensemble learning** to support climate adaptation and coastal risk assessment in low-lying reef islands.  

## Author
**Yiyu Yang**  
MSc in Environmental Data Science and Machine Learning  
Imperial College London  

Supervisors: Dr. Yves Plancherel, Prof. Matthew Piggott  

## How to Run
See [`codes/README.md`](./codes/README) for setup, dependencies, and execution instructions.  
