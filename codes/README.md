# Reef Island Shoreline Modeling with Machine Learning

Reef islands in the Maldives are low-lying accumulations of biogenic carbonate sediment whose shorelines respond dynamically to interacting oceanographic, climatic, and geomorphic drivers. Predicting shoreline change for natural reef islands is essential for risk-informed coastal planning and climate adaptation.  

This repository develops and evaluates **nine machine learning approaches** to predict along-transect shoreline positions using multi-source predictors and a novel **hybrid transect design** that minimizes orientation bias.

---

## 🚀 Project Overview
- **Objective:** Predict shoreline change of reef islands using machine learning.  
- **Methods:** Ensemble learning, regression models, neural networks.  
- **Data Sources:** Sentinel-2 & Landsat imagery (via CoastSat), ECMWF reanalysis (wind & wave), reef geomorphometrics.  
- **Study Area:** Natural reef islands in the South Maalhosmadulu Atoll, Maldives (2016–2024).  

---

## ⚙️ Environment Setup
This project is based on [CoastSat](https://github.com/kvos/CoastSat). Please follow CoastSat’s installation guide (`CoastSat/README.md`) before proceeding.  

Clone this repository and create the environment:
```bash
cd ./codes
conda env create -f environment.yml
conda activate shoreline_ml
```
> ⚠️ This automatic installation process may take more than 10 minutes to complete all package installations.

The `CoastSat` folder in this repository contains modified source code adapted for reef island applications.  

---

## 📑 Workflow

### 1. Shoreline Data Collection

- Use **`CoastSat/Console_panel.ipynb`** to download and process shoreline positions of target islands from **2016–2024**.  
- This procedure includes approximately **10–20 minutes of manual labeling**, which is a required step in the **CoastSat** workflow.  
- If you wish to skip this step, please download the **`Dhakendhoo`** folder from the provided **OneDrive link** ([link](https://imperiallondon-my.sharepoint.com/:f:/g/personal/yy1824_ic_ac_uk/EuO8DvfJc45Ch39DztyfKRYB1BzEZ2FsaDL0TceSNnL3GQ?e=Hv6chn)), place it in the **`CoastSat/data`** directory, and then bypass the data download and manual labeling stages when running **`CoastSat/Console_panel.ipynb`**.  
- When performing *tidal correction* in **`Console_panel.ipynb`**, you will also need the **`load_tide`** and **`ocean_tide_20241025`** folders, which are available from the same OneDrive link. Ensure that the file paths specified in **`CoastSat/fes2022.yaml`** are consistent with the actual storage locations of these models. For details, please refer to the *tidal correction* instructions in **`Console_panel.ipynb`**.  
- After tidal correction, the processed transects will be saved in the corresponding subfolders under **`CoastSat/data/`** as **`transect_time_series_tidally_corrected.csv`**.  
- If you prefer to avoid this step, simply download the **`Dhakendhoo`** folder, place it in **`CoastSat/data`**, and proceed directly to the next step using the provided pre-processed shoreline dataset for Dhakendhoo as an example.  



> ⚠️ If you prefer to skip processing, download the **`Dhakendhoo`** folder from the OneDrive link provided above, place it in the **`CoastSat/data`** directory, and proceed directly to the next step using the provided Dhakendhoo data as an example.  


---

### 2. Data Preparation
- Use notebooks in the `Fetch_data/` folder to assemble predictor variables:  
- `prepare_data.ipynb`  
- `Reef_Geomorphometrics.ipynb`  
- `ECMWF.ipynb`  
- For testing, `prepare_data.ipynb` contains pre-configured code to run directly.  

---

### 3. Model Training & Prediction
- Run **`Modelling/notebook/run_models.ipynb`** to train and evaluate nine models:  
- **Ensemble methods:** Random Forest, LightGBM, XGBoost, Stacking  
- **Linear model:** Ridge Regression  
- **Instance-based method:** K-Nearest Neighbors  
- **Neural networks:** Multi-layer Perceptron (MLP), feed-forward neural networks, auto-tuned variants  

The notebook is fully configured for direct execution.  

---

## 📝 Notes
- `Console_panel.ipynb`, `prepare_data.ipynb`, `Reef_Geomorphometrics.ipynb`, and `ECMWF.ipynb` must be run **separately for each island**.  
- `run_models.ipynb` supports **multi-island training and prediction** in a single run.  
- Large external datasets (e.g., tidal models, pre-processed shoreline data) are **not included** due to size and copyright restrictions.  

---

## 📖 Citation
If you use this repository in your research, please cite appropriately (citation details will be added upon publication).  

---

## 📂 License & Data Use
- Source code is open for research use (license details to be confirmed).  
- Some large datasets and third-party products (e.g., ECMWF, satellite imagery) may require separate access rights or licenses.  

