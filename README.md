# Data-and-Machine-Learning-Encoding

This repository demonstrates **7 essential encoding techniques** for preparing categorical and cyclic data for machine learning models. Every method is illustrated using or inspired by features from the **Ames Housing dataset**.

---

## Table of Contents
1. [Label Encoding](#method-1-label-encoding)
2. [One-Hot Encoding](#method-2-one-hot-encoding)
3. [Feature Hashing](#method-3-feature-hashing)
4. [Dataset Statistics Encoding](#method-4-encoding-categories-with-dataset-statistics)
5. [Cyclic Features Encoding](#cyclic-features-encoding-cyclic-features)
6. [Target Encoding](#method-5-target-encoding)
7. [K-Fold Target Encoding](#method-6-k-fold-target-encoding)

---

## Method 1: Label Encoding
Assigns a unique integer to each category based on alphabetical order or custom ranking.
* **Best Used For:** Ordinal features (categories with an inherent order).
* **Ames Housing Example:** `KitchenQual` (Ex, Gd, TA, Fa, Po).
* **Pros:** Keeps dataset memory footprint low; preserves natural order.
* **Cons:** Introduces a false mathematical hierarchy for non-ordinal features (e.g., Red = 1, Blue = 2).

## Method 2: One-Hot Encoding
Creates a new binary column ($0$ or $1$) for each unique category in a feature.
* **Best Used For:** Nominal features with low-to-medium cardinality (few unique values).
* **Ames Housing Example:** `Foundation` (PConc, CBlock, BrkTil, Wood).
* **Pros:** Works perfectly with linear models; introduces no false numerical ranking.
* **Cons:** Causes the "curse of dimensionality" on high-cardinality features.

## Method 3: Feature Hashing
Uses a hashing function to map high-cardinality categorical values into a fixed-size numerical vector.
* **Best Used For:** High-cardinality features or streaming data where vocabulary size is unknown.
* **Ames Housing Example:** `Neighborhood` or `Exterior1st`.
* **Pros:** Extreme memory efficiency; handles new or unseen categories automatically.
* **Cons:** Irreversible transformation; potential "hash collisions" where different categories get the same ID.

## Method 4: Encoding Categories with Dataset Statistics
Replaces categories with frequency counts or percentages calculated directly from the dataset.
* **Best Used For:** High-cardinality features where frequency correlates with the target variable.
* **Ames Housing Example:** `SaleType` or `MSZoning`.
* **Pros:** Easy to compute; preserves representation weight without adding new columns.
* **Cons:** Destroys category identity if two entirely different categories share the exact same frequency count.

## Cyclic Features: Encoding Cyclic Features
Transforms cyclical continuous features into coordinates using **Sine and Cosine transformations**.
* **Best Used For:** Time-based, seasonal, or repeating numerical patterns.
* **Ames Housing Example:** `MoSold` (Month Sold, 1-12).
* **Pros:** Ensures the model understands that January ($1$) and December ($12$) are chronologically adjacent.
* **Cons:** Doubles the feature column count; can confuse non-tree-based algorithms if not scaled.

## Method 5: Target Encoding
Replaces each category value with the global expected mean of the target variable for that specific category.
* **Best Used For:** High-cardinality categorical features used in tree-based algorithms.
* **Ames Housing Example:** Encoding `Neighborhood` using the mean `SalePrice`.
* **Pros:** Builds a direct, strong linear relationship between the category and the target.
* **Cons:** High risk of **target leakage** and severe overfitting if not heavily smoothed or regularised.

## Method 6: K-Fold Target Encoding
An advanced version of target encoding that calculates the category target means out-of-fold using a cross-validation split.
* **Best Used For:** Preventing overfitting in production-grade gradient boosting models (XGBoost, LightGBM).
* **Ames Housing Example:** Encoding `Neighborhood` relative to `SalePrice` inside a 5-Fold split.
* **Pros:** Mitigates target leakage; vastly improves model generalization on unseen data.
* **Cons:** Computationally complex to track, build, and deploy to pipeline pipelines.

---

##  Requirements & Installation
Ensure you have the following libraries installed before running the project notebooks:
```bash
pip install numpy pandas scikit-learn category_encoders
```
