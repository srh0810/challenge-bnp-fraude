# BNP Paribas Personal Finance - Fraud Detection Challenge

## Overview

This project implements a complete fraud detection solution for shopping basket data from BNP Paribas Personal Finance. The goal is to identify fraudulent transactions using machine learning techniques applied to customer shopping basket information.

**Objective**: Maximize PR-AUC score (target > 0.14)

## Problem Context

BNP Paribas Personal Finance, Europe's #1 consumer credit provider, faces increasing fraud challenges. Fraudsters continuously evolve their methods to normalize their behavior and make it difficult to detect. The main challenge is the low occurrence of fraud in the population (~1.4% fraud rate), making this a highly imbalanced classification problem.

## Dataset Description

### Data Structure
- **Total size**: 115,988 observations, 147 columns
- **Training set**: 92,790 observations (1,319 fraud cases)
- **Test set**: 23,198 observations (362 fraud cases)
- **Fraud rate**: ~1.4%

### Features
Each observation represents a shopping basket with up to 24 items. For each item (1-24), we have:
- `item{i}`: Product category (e.g., "Computer")
- `cash_price{i}`: Item price
- `make{i}`: Brand/manufacturer (e.g., "Apple")
- `model{i}`: Product model description
- `goods_code{i}`: Item code
- `Nbr_of_prod_purchas{i}`: Quantity purchased
- `Nb_of_items`: Total number of different items in basket

### Target Variable
- `fraud_flag`: Binary indicator (1 = fraud, 0 = legitimate)

## Data Setup

### Prerequisites
Download the challenge data from: https://challengedata.ens.fr/participants/challenges/104/

### File Structure
```
project/
├── data/
│   ├── X_train.csv
│   ├── Y_train.csv
│   └── X_test.csv
├── fraud_detection.ipynb
└── README.md
```

Place the downloaded files in a `data/` folder at the project root with these exact names:
- `X_train.csv` (training features)
- `Y_train.csv` (training labels) 
- `X_test.csv` (test features)

## Methodology

### Feature Engineering
The solution implements comprehensive feature engineering including:

1. **Basic Basket Features**
   - Total basket price
   - Total number of products
   - Average item price

2. **Category & Brand Analysis**
   - Number of unique categories/brands
   - Diversity ratios
   - Brand concentration metrics

3. **Electronics-Specific Features**
   - Electronics detection flags
   - Electronics price ratios
   - Electronics count

4. **Statistical Features**
   - Price statistics (mean, std, skewness)
   - Quantity statistics
   - Price concentration ratios

5. **Behavioral Indicators**
   - Basket size flags (small, medium, large)
   - Expensive basket indicators
   - Brand loyalty indicators

### Models Used
The solution employs an ensemble approach with multiple algorithms:

1. **XGBoost** - Gradient boosting for complex patterns
2. **LightGBM** - Fast gradient boosting (primary model)
3. **Random Forest** - Tree-based ensemble
4. **Logistic Regression** - Linear baseline with scaled features

### Evaluation Metric
- **PR-AUC (Precision-Recall Area Under Curve)**: Optimal for imbalanced datasets
- Equivalent to `sklearn.metrics.average_precision_score`

## Usage

1. **Setup Environment**
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
   ```

2. **Download Data**
   - Get files from ENS challenge website
   - Place in `data/` folder with correct names

3. **Run Analysis**
   ```python
   # Load and execute the notebook
   jupyter notebook fraud_detection.ipynb
   ```

4. **Output**
   - Model performance metrics via cross-validation
   - Final predictions saved to `submission.csv`

## Key Results

### Expected Performance
- **Benchmark 1** (Random): PR-AUC = 0.017
- **Benchmark 2** (ML optimized): PR-AUC = 0.14
- **Target**: PR-AUC > 0.14

### Model Insights
- Large baskets (>10 items) show higher fraud rates
- Electronics are common but have lower fraud rates
- Price concentration and brand diversity are key indicators
- Basket size and composition patterns are strong predictors

## Technical Implementation

### Code Structure
- **Centralized feature engineering** function to avoid duplication
- **Consistent preprocessing** between train/test sets
- **Cross-validation** for robust model evaluation
- **Class weight balancing** for imbalanced data
- **Ensemble prediction** using best performing model (LightGBM)

### Key Features
- Handles missing values appropriately
- Scales features for logistic regression
- Uses stratified sampling for validation
- Implements proper train/test data leakage prevention

## Files Generated

- `submission.csv`: Final predictions for test set (ID, fraud_flag probability)

## Notes

- The solution prioritizes recall while maintaining precision due to the fraud detection context
- Feature engineering is the most critical component for performance
- LightGBM typically provides the best balance of speed and accuracy
- Cross-validation ensures robust performance estimates

## Contact

For questions about the implementation or methodology, please refer to the challenge documentation at ENS or review the detailed comments in the notebook code.
