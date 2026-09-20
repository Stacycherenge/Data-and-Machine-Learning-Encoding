# Data-and-Machine-Learning-Encoding
## House Prices — Encoding Techniques & Regression

A walkthrough of preparing the Kaggle House Prices: Advanced Regression Techniques dataset for modeling. This pipeline focuses on matching the correct encoding technique to each specific categorical variable type rather than using a single method for everything.

## Contents
* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Repository Structure](#repository-structure)
* [Setup & Installation](#setup--installation)
* [Methodology](#methodology)
* [Results](#results)
* [Key Design Decisions](#key-design-decisions)

## Project Overview
The goal is to predict residential property `SalePrice` in Ames, Iowa using ~80 features. This dataset serves as an ideal testbed for tailored categorical encoding:

* **Ranked features** (e.g., `ExterQual`: Poor → Excellent) require preserving order.
* **Unordered, low-cardinality features** (e.g., `RoofStyle`) require nominal treatment without inflating the feature space.
* **Unordered, high-cardinality features** (e.g., `Neighborhood`, 25 categories) need a treatment that avoids exploding the column count.
* **Secret categories** (e.g., `MSSubClass`) look numeric but are actually codes.
* **Cyclic features** (e.g., `MoSold`) repeat periodically.

## Dataset
* **Source:** Kaggle — House Prices Competition
* **Training Data:** `train.csv` (1,460 rows, 81 columns)
* **Test Data:** `test.csv` (1,459 rows, 80 columns)
* **Target Variable:** `SalePrice` (continuous, USD)

## Repository Structure
```text
.
├── House_prices.ipynb        
├── data_description.txt      
├── train.csv                 
├── test.csv                
├── sample_submission.csv     

├── requirements.txt          
└── README.md
```

## Setup & Installation

**1. Clone and enter repository**
```bash
git clone <repo-url>
cd Data-and-Machine-Learning-Encoding
```

**2. Environment setup**
```bash
python -m venv venv
source venv/bin/activate        
pip install -r requirements.txt
```

**3. Launch environment**
```bash
jupyter notebook House_prices.ipynb
```

> **Note:** `train.csv`, `test.csv`, and `data_description.txt` are already included in this repository — no separate download needed.

## Methodology

### 1. Data Type Corrections
* **MSSubClass Casting:** Converted from integer to string (`str`) to prevent models from assuming false numerical relationships between dwelling type codes.

### 2. Validation Strategy
* **Train/Val Split:** 80/20 split using `random_state=42`. Official test data remains completely untouched during feature engineering.

### 3. Missing Value Imputation
* **Structural Absence (NA = feature missing):** Replaced with `"None"` (categorical) or `0` (numerical fields like `GarageYrBlt`) to signify the house lacks that asset.
* **Genuine Missingness:** Imputed using training-only mode (categorical) or median (numerical) to prevent validation leakage.
* **Test-only gaps:** A small number of values missing only in the official `test.csv` (columns complete in `train.csv`) are caught with a fallback, also fit on `X_train` only.

### 4. Categorical Encoding Strategy
Four distinct encoding techniques, each matched to what the variable actually needs:

| Method | Applied to | Why |
| :--- | :--- | :--- |
| **Ordinal Encoding** | Ranked quality/condition columns (e.g., `ExterQual`, `BsmtFinType1`, `LotShape`) | Explicit integer mappings preserve the real order (Poor → Excellent) that one-hot encoding would discard. |
| **One-Hot Encoding** | Unordered, low-cardinality columns (e.g., `RoofStyle`, `Foundation`, `SaleCondition`) | Handled via scikit-learn's `OneHotEncoder(handle_unknown='ignore')`, fit strictly on training data so validation/test columns always match. |
| **Frequency Encoding** | Unordered, high-cardinality columns (`MSSubClass`, `Neighborhood`, `Exterior1st`, `Exterior2nd`) | One-hot encoding these four alone would add 71 sparse columns. Instead, each category is replaced with how often it occurs in `X_train` (as a proportion) — one dense numeric column instead of dozens of binary ones. |
| **Cyclic Encoding** | `MoSold` | Transformed (1–12) into sine/cosine pairs to properly connect December (12) and January (1), which are adjacent in reality but maximally far apart as plain integers. |

### 5. Modeling
Evaluated continuous predictions using RMSE and R² across two architectures:
* **Linear Regression:** baseline model.
* **Random Forest Regressor:** ensemble of 300 trees, to capture non-linear feature combinations.

## Results

| Model | Validation RMSE | Validation R² |
| :--- | :--- | :--- |
| Linear Regression (Baseline) | \$30,624 | 0.878 |
| **Random Forest Regressor** | **\$28,533** | **0.894** |

The Random Forest model achieved superior performance, indicating that non-linear feature interactions (such as quality weighting varying across different neighborhoods) are relevant in this dataset. Introducing frequency encoding for the four highest-cardinality columns also reduced the total feature count substantially (from ~244 columns under a pure one-hot approach down to ~177) without any loss in validation performance.

## Key Design Decisions
* **Context-Driven Imputation:** Separated structural "no feature" gaps from completely missing observations to preserve clean signal distributions.
* **Cardinality-Aware Encoding:** Rather than applying one-hot encoding uniformly, high-cardinality nominal columns were routed to frequency encoding instead, trading a small amount of category-level distinctiveness for a much more compact and less sparse feature matrix.
* **Strict Training Isolation:** Every calculation (medians, modes, encoder mappings, frequency tables) was fit exclusively on `X_train` to eliminate silent data leakage into validation or test data.
* **Enforced Matrix Symmetry:** Leveraged scikit-learn's `OneHotEncoder` over `pd.get_dummies` to guarantee perfectly matching validation/test shapes.
