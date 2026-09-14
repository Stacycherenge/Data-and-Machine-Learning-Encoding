# Data-and-Machine-Learning-Encoding
Here is a streamlined, highly scannable version of your README. It trims down the heavy text blocks into punchy fragments while keeping all the critical technical details intact.
------------------------------
## House Prices — Encoding Techniques & Regression
A walkthrough of preparing the Kaggle House Prices: Advanced Regression Techniques dataset for modeling. This pipeline focuses on matching the correct encoding technique to each specific categorical variable type rather than using a single method for everything.
## Contents

* Project Overview
* Dataset
* Repository Structure
* Setup & Installation
* Methodology
* Results
* Key Design Decisions

------------------------------
## Project Overview
The goal is to predict residential property SalePrice in Ames, Iowa using ~80 features. This dataset serves as an ideal testbed for tailored categorical encoding:

* Ranked features (e.g., ExterQual: Poor → Excellent) require preserving order.
* Unordered features (e.g., Neighborhood) require nominal treatment.
* Secret categories (e.g., MSSubClass) look numeric but are actually codes.
* Cyclic features (e.g., MoSold) repeat periodically.

------------------------------
## Dataset

* Source: [Kaggle — House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
* Training Data: train.csv (1,460 rows, 81 columns)
* Test Data: test.csv (1,459 rows, 80 columns)
* Target Variable: SalePrice (Continuous, USD)

------------------------------
## Repository Structure

.
├── House_prices_corrected.ipynb   # Full cleaning, encoding, and modeling notebook
├── data_description.txt           # Kaggle category code reference definitions
├── requirements.txt               # Pinpinned dependencies (pandas 3.0, scikit-learn 1.8)
└── README.md

------------------------------
## Setup & Installation

# Clone and enter repository
git clone <your-repo-url> && cd <your-repo-folder>
# Environment setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Launch environment
jupyter notebook House_prices_corrected.ipynb

Note: Download train.csv, test.csv, and data_description.txt directly from Kaggle and place them in the root directory before running.
------------------------------
## Methodology## 1. Data Type Corrections

* MSSubClass Casting: Converted from integer to string (str) to prevent models from assuming false numerical relationships between dwelling type codes.

## 2. Validation Strategy

* Train/Val Split: 80/20 data split using random_state=42. Official test data remains completely untouched during engineering.

## 3. Missing Value Imputation

* Structural Absence (NA = Feature Missing): Replaced with "None" (categorical) or 0 (numerical fields like GarageYrBlt) to signify the house lacks that asset.
* Genuine Missingness: Imputed using training-only mode (categorical) or median (numerical) to prevent validation leakage.

## 4. Categorical Encoding Split

* Ordinal Encoding: Applied to ranked quality/condition columns using explicit integer mappings.
* One-Hot Encoding: Applied to unordered fields (e.g., RoofStyle). Handled via Scikit-Learn’s OneHotEncoder(handle_unknown='ignore') fit strictly on training data.
* Cyclic Encoding: Transformed MoSold (1–12) into Sine/Cosine pairs to properly connect December (12) and January (1).

## 5. Modeling

* Evaluated continuous predictions using RMSE and R² across two architectures:
* Linear Regression: Baseline model.
   * Random Forest Regressor: Complex ensemble (300 trees) to capture non-linear feature combinations.

------------------------------
## Results

| Model | Validation RMSE | Validation R² |
|---|---|---|
| Linear Regression (Baseline) | $30,624 | 0.878 |
| Random Forest Regressor | $28,533 | 0.894 |

Takeaway: The Random Forest model achieved superior performance, indicating that non-linear feature interactions (such as quality weight variations across different neighborhoods) are highly relevant in this dataset.
------------------------------
## Key Design Decisions

* Context-Driven Imputation: Separated structural "no feature" gaps from completely missing observations to preserve clean signal distributions.
* Strict Training Isolation: Every calculation (medians, modes, encoder mappings) was fit exclusively on X_train to eliminate silent data leakage.
* Enforced Matrix Symmetry: Leveraged Scikit-Learn pipelines over pd.get_dummies to guarantee perfectly matching validation/test shapes.

------------------------------
Would you like me to help you expand the missing Limitations or License sections at the bottom?

