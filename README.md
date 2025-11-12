# 🚦 Road Accident Analysis: EDA, Clustering and Classification of High-Risk Areas

## Project Overview

This project explores and analyzes **road traffic accidents in the United States** using the **U.S. Accidents dataset (2016–2023)**.  
It combines **Exploratory Data Analysis (EDA)**, **unsupervised learning (clustering)**, and **supervised learning (classification)** to uncover insights and predict accident severity.

### Main Objectives
-  **EDA:** Understand accident trends across time, geography, and weather conditions.  
-  **Classification:** Predict accident severity based on environmental and temporal factors.
-  **Pattern Discovery: Clustering** Identify high-risk accident areas based on environmental and temporal factors.


---

##  Dataset

- **Source:** [Kaggle – U.S. Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)  
- **Size:** ~7 million records across 49 U.S. states  
- **Subset used:** 2020–2022 (to optimize processing)
- **Features:** 46 variables including:
  - Temporal: accident date/time  
  - Spatial: latitude, longitude, state, city  
  - Environmental: weather, visibility, precipitation  
  - Infrastructure: signal, crossing, distance  
  - Target variable: `Severity` (1–4)


##  Exploratory Data Analysis (EDA)

The EDA phase aimed to understand accident dynamics and uncover hidden patterns.

### Key Findings:
-  **Top states:** California, Texas, and Florida have the highest number of accidents.  
-  **Seasonal patterns:** Higher accident rates during winter months.  
-  **Time patterns:** Two daily peaks — 7–9 AM and 4–6 PM.  
-  **Weather:** Most accidents occur under cloudy or overcast conditions.  
-  **Severity imbalance:** Mild accidents dominate; severe ones are rare but critical.

Visualizations include:
- Heatmaps by state and city  
- Temporal distributions (month, day, hour)  
- Severity histograms and correlation matrices  


##  Data Preprocessing
- Managed missing values using mean/median imputation  
- Removed constant or irrelevant features (e.g., `ID`, `Country`)  
- Encoded categorical variables (One-Hot and Frequency Encoding)  
- Reduced dimensionality by filtering low-correlation features  
- Balanced data using **SMOTE** and **ADASYN**


##  Classification (Severity Prediction)
| Model | Accuracy | F1-Score | Notes |
|-------|-----------|----------|-------|
| KNN | 82% | 0.35 | Weak for minority classes |
| Random Forest | 90% | 0.42 | High interpretability |
| LightGBM | 90% | 0.43 | Efficient on large data |
| XGBoost | 91% | 0.44 | Best global performance |

After applying **SMOTE** for class imbalance:
- XGBoost + SMOTE → Accuracy: 0.88, F1-Score: 0.50


##  Clustering Analysis

Applied to **California accidents (2022)** to identify high-risk areas.

| Algorithm | Key Findings |
|------------|---------------|
| **K-Means (k=14)** | Found clusters linked to urban vs rural zones, weather and time. Cluster 6: 61% severe accidents. |
| **DBSCAN** | Detected dense vs. isolated regions; highlighted rainy, nighttime conditions. |
| **Agglomerative** | Grouped similar accident profiles by severity and visibility. |


## Hybrid Approach
Combining clustering and classification improved results:
- **Accuracy:** 0.91  
- **F1-score:** 0.82 (with XGBoost + Optuna optimization)
  
After re-grouping severity levels:  
- **Mild:** (Severity 1–2)  
- **Severe:** (Severity 3–4)


##  Key Insights

- **High-risk clusters** correspond to rainy, low-visibility regions.
- **Severity 4 accidents** often occurred under poor weather or lighting conditions.
- Binary classification (Sev1–2 vs. Sev3–4) yielded clearer predictive patterns than 4-class classification.


##  Reference Papers
- Rezashoar, S. et al. (2024) – *LightGBM-Optuna for Accident Severity Classification*  
- Khosravi, Y. et al. (2024) – *Accident Prone Areas via ML and Spatial Analysis*  
- Sinclair, C. & Das, S. (2021) – *Traffic Accident Analytics in UK Urban Areas*

##  Tools & Libraries

- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, Optuna, Dask  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Environment:** Jupyter / Google Colab  


