# Energy Consumption Modelling (EIA)

This project analyzes and models electricity fuel consumption using publicly available operational data from the **U.S. Energy Information Administration (EIA)**.  
The objective is not only prediction, but understanding **where predictive signal comes from** in power system data.

The analysis compares linear and non-linear models and explicitly studies the impact of including and excluding electricity generation as a feature.

---

## Dataset

The dataset is sourced from the **U.S. EIA Electricity Power Operational Data API** and contains annual, aggregated power system statistics across regions, sectors, and fuel types.

**Key features include:**
- Electricity generation (MWh)
- Fuel consumption (MMBtu)
- Fuel type
- Sector (electric utility, etc.)
- Regional/state-level aggregation

**Target variable:**
- Total fuel consumption (MMBtu)

---

## Methodology

### 1. Data Extraction
- Data is programmatically fetched from the EIA API and written to a CSV file.
- The raw dataset is cleaned and structured for analysis.

### 2. Feature Engineering
- Categorical features (state, sector, fuel type) are one-hot encoded.
- Redundant features (e.g., location vs state) are removed.

### 3. Correlation Analysis
- Feature correlations with fuel consumption are examined.
- Electricity generation shows a dominant linear relationship with fuel consumption.

### 4. Modeling Approach
- **Ridge Regression** is used as a regularized linear baseline.
- **Random Forest Regression** is used to capture potential non-linear interactions.
- Models are evaluated **with and without the generation feature** to isolate structural signal.

### 5. Validation
- Grid search is used for Random Forest as a validation step to confirm trends rather than aggressively optimize performance.

---

## Key Insight

Fuel consumption is strongly governed by electricity generation due to physical energy conversion relationships:

\[
Fuel\ Consumption\ (MMBtu) = Generation\ (MWh) \times \frac{Heat\ Rate\ (Btu/kWh)}{1000}
\]

When generation is included, both linear and non-linear models achieve high R² scores.  
When generation is removed, performance drops sharply, revealing that remaining features (fuel type, sector, region) carry **limited but non-zero structural information**.

Random Forest models show a modest improvement over Ridge regression, suggesting mild non-linear interactions, but physics-driven relationships dominate overall behavior.

---

## Results Summary

| Model            | Generation Included | R² Score |
|------------------|---------------------|----------|
| Ridge Regression | Yes                 | ~0.96    |
| Ridge Regression | No                  | ~0.26    |
| Random Forest    | Yes                 | ~0.97    |
| Random Forest    | No                  | ~0.36    |

---

## Repository Structure

- `energy_consumption_modelling.ipynb` — exploratory analysis and reasoning
- `energy_consumption_modelling.py` — reproducible script version
- `electricity-data.csv` — processed dataset
- `correlation_heatmap.png` — feature correlation visualization
- `eia_data_extraction.js` — EIA API data extraction script

---

## License

This project is released under the MIT License.
