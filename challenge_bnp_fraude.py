# -*- coding: utf-8 -*-
"""
BNP Paribas Personal Finance - Fraud Detection Challenge
Complete solution for fraud detection on shopping basket data
Objective: Maximize PR-AUC (> 0.14)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import average_precision_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===============================
# GLOBAL CONSTANTS
# ===============================

N_ITEMS = 24
RANDOM_STATE = 42
ELECTRONICS = ['Computer', 'Phone', 'TV']
COMMON_ITEMS = ['COMPUTERS', 'TELEVISIONS HOME CINEMA', 'COMPUTER PERIPHERALS ACCESSORIES', 'PHONES', 'TABLETS']
COMMON_MAKES = ['APPLE', 'SAMSUNG', 'SONY', 'LG', 'HP']

# Column names
PRICE_COLS = [f'cash_price{i}' for i in range(1, N_ITEMS + 1)]
QTY_COLS = [f'Nbr_of_prod_purchas{i}' for i in range(1, N_ITEMS + 1)]
ITEM_COLS = [f'item{i}' for i in range(1, N_ITEMS + 1)]
MAKE_COLS = [f'make{i}' for i in range(1, N_ITEMS + 1)]

# ===============================
# DATA LOADING
# ===============================

DATA_PATH = "data/"

x_train = pd.read_csv(f"{DATA_PATH}X_train.csv", low_memory=False)
y_train = pd.read_csv(f"{DATA_PATH}Y_train.csv", low_memory=False)
x_test  = pd.read_csv(f"{DATA_PATH}X_test.csv", low_memory=False)


print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)

# ===============================
# EXPLORATORY DATA ANALYSIS - EDA
# ===============================

# Merge data for EDA
train_df = pd.concat([x_train, y_train], axis=1)
fraud_rate = train_df['fraud_flag'].mean()
print(f"Global fraud rate: {fraud_rate:.3%}")

# Distribution of number of items per basket
sns.countplot(x='Nb_of_items', data=train_df)
plt.title("Distribution of number of items per basket")
plt.show()

# Fraud rate by number of items
fraud_by_items = train_df.groupby('Nb_of_items')['fraud_flag'].agg(['count', 'sum', 'mean'])
print(fraud_by_items)

# ===============================
# FEATURE ENGINEERING
# ===============================

def calculate_features(df):
    """
    Calculate all features for a given DataFrame.
    Centralized function to avoid code duplication.
    """
    
    # ---- Basic basket features ----
    df['total_basket_price'] = 0
    df['total_products'] = 0
    
    for price_col, qty_col in zip(PRICE_COLS, QTY_COLS):
        df['total_basket_price'] += df[price_col].fillna(0) * df[qty_col].fillna(0)
        df['total_products'] += df[qty_col].fillna(0)
    
    df['avg_item_price'] = df['total_basket_price'] / df['Nb_of_items'].replace(0, 1)
    
    # ---- Category and brand diversity ----
    df['unique_categories'] = df[ITEM_COLS].nunique(axis=1)
    df['unique_makes'] = df[MAKE_COLS].nunique(axis=1)
    
    # ---- Electronics features ----
    df['has_electronics'] = 0
    df['electronics_count'] = 0
    df['electronics_price'] = 0
    
    for i in range(1, N_ITEMS + 1):
        is_elec = df[f'item{i}'].isin(ELECTRONICS).astype(int)
        df['has_electronics'] |= is_elec
        df['electronics_count'] += is_elec * df[f'Nbr_of_prod_purchas{i}'].fillna(0)
        df['electronics_price'] += is_elec * df[f'cash_price{i}'].fillna(0) * df[f'Nbr_of_prod_purchas{i}'].fillna(0)
    
    # ---- Additional ratios and indicators ----
    df['avg_qty_per_item'] = df['total_products'] / df['Nb_of_items'].replace(0, 1)
    df['price_per_product'] = df['total_basket_price'] / df['total_products'].replace(0, 1)
    df['make_diversity_ratio'] = df['unique_makes'] / df['Nb_of_items'].replace(0, 1)
    df['category_diversity_ratio'] = df['unique_categories'] / df['Nb_of_items'].replace(0, 1)
    
    # ---- Price statistics ----
    df['max_item_price'] = df[PRICE_COLS].max(axis=1)
    df['min_item_price'] = df[PRICE_COLS].min(axis=1)
    df['std_item_price'] = df[PRICE_COLS].std(axis=1)
    df['skew_item_price'] = df[PRICE_COLS].skew(axis=1)
    
    # ---- Concentration ratios ----
    df['max_price_ratio'] = df['max_item_price'] / df['total_basket_price'].replace(0, 1)
    df['electronics_ratio'] = df['electronics_price'] / df['total_basket_price'].replace(0, 1)
    
    # ---- Flags ----
    df['make_diversity_per_product'] = df['unique_makes'] / df['total_products'].replace(0, 1)
    df['category_diversity_per_product'] = df['unique_categories'] / df['total_products'].replace(0, 1)
    df['is_large_basket'] = (df['Nb_of_items'] > 10).astype(int)
    df['is_single_brand'] = (df['unique_makes'] == 1).astype(int)
    
    # ---- Additional features ----
    # Basket size flags
    df['large_basket_flag'] = (df['Nb_of_items'] > 5).astype(int)
    df['medium_basket_flag'] = ((df['Nb_of_items'] > 1) & (df['Nb_of_items'] <= 5)).astype(int)
    
    # Electronics ratios
    df['avg_electronics_price'] = df['electronics_price'] / df['electronics_count'].replace(0,1)
    df['elec_ratio'] = df['electronics_count'] / df['total_products'].replace(0,1)
    
    # Diversity ratios
    df['make_ratio'] = df['unique_makes'] / df['total_products'].replace(0,1)
    df['category_ratio'] = df['unique_categories'] / df['total_products'].replace(0,1)
    
    # Item statistics
    df['num_items_nonnull'] = df[PRICE_COLS].notnull().sum(axis=1)
    
    # Quantity statistics
    df['avg_qty'] = df[QTY_COLS].replace(np.nan, 0).mean(axis=1)
    df['max_qty'] = df[QTY_COLS].replace(np.nan, 0).max(axis=1)
    df['min_qty'] = df[QTY_COLS].replace(np.nan, 0).min(axis=1)
    df['std_qty'] = df[QTY_COLS].replace(np.nan, 0).std(axis=1)
    
    # Top 3 items analysis
    df['top3_unique_makes'] = df[[f'make{i}' for i in range(1,4)]].nunique(axis=1)
    df['top3_unique_categories'] = df[[f'item{i}' for i in range(1,4)]].nunique(axis=1)
    df['top3_make_ratio'] = df['top3_unique_makes'] / df['unique_makes'].replace(0,1)
    df['top3_category_ratio'] = df['top3_unique_categories'] / df['unique_categories'].replace(0,1)
    
    # Apple flag
    apple_count = sum([(df[f'make{i}']=='APPLE').astype(int) for i in range(1, N_ITEMS + 1)])
    df['apple_flag'] = (apple_count >= 2).astype(int)
    
    # Categorical encoding for top items and makes
    for i in range(1, 4):
        item_col = f'item{i}'
        make_col = f'make{i}'
        
        if item_col in df.columns:
            for item in COMMON_ITEMS:
                df[f'has_{item.lower().replace(" ", "_")}_item{i}'] = (df[item_col] == item).astype(int)
            
            for make in COMMON_MAKES:
                df[f'has_{make.lower().replace(" ", "_")}_make{i}'] = (df[make_col] == make).astype(int)
    
    return df

# Apply feature engineering to training data
print("Applying feature engineering to training data...")
train_df = calculate_features(train_df)

# Calculate price threshold from training data
price_thresh = train_df['total_basket_price'].quantile(0.9)
train_df['is_expensive_basket'] = (train_df['total_basket_price'] > price_thresh).astype(int)

print("Basic basket analysis:")
print("Average price fraudulent basket:", train_df[train_df['fraud_flag']==1]['total_basket_price'].mean())
print("Average price normal basket:", train_df[train_df['fraud_flag']==0]['total_basket_price'].mean())

# ===============================
# MODEL PREPARATION
# ===============================

# Select final features
FEATURE_COLS = [
    'total_basket_price', 'total_products', 'avg_item_price',
    'has_electronics', 'electronics_count', 'electronics_price',
    'unique_makes', 'unique_categories',
    'avg_qty_per_item', 'price_per_product',
    'make_diversity_ratio', 'category_diversity_ratio',
    'max_item_price', 'min_item_price', 'std_item_price',
    'max_price_ratio', 'electronics_ratio',
    'is_expensive_basket', 'is_large_basket', 'is_single_brand',
    'Nb_of_items'
]

# Check available features
available_features = [col for col in FEATURE_COLS if col in train_df.columns]
print(f"Available features: {len(available_features)}")

# Prepare X and y
X = train_df[available_features].fillna(0)
y = train_df['fraud_flag']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ===============================
# MODEL TRAINING
# ===============================

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", weight_dict)

# Define models
models = {
    "XGBoost": xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        scale_pos_weight=class_weights[1]/class_weights[0],
        random_state=RANDOM_STATE,
        use_label_encoder=False
    ),
    "Random Forest": RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver='liblinear'
    ),
    "LightGBM": lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight_dict[0]/weight_dict[1],
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# Cross-validation and training
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Choose appropriate data based on model
    X_use = X_scaled if name == "Logistic Regression" else X.values
    
    print(f"  X_use shape: {X_use.shape}")
    print(f"  y shape: {y.shape}")

    # Cross-validation PR-AUC
    cv_scores = cross_val_score(model, X_use, y, cv=cv, scoring='average_precision')
    results[name] = cv_scores
    print(f"   PR-AUC CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Final training on all data
    model.fit(X_use, y)
    models[name] = model

print("\nCV PR-AUC Results:")
for name, scores in results.items():
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# ===============================
# TEST DATA PROCESSING
# ===============================

print("\nApplying feature engineering to test data...")
test_df = x_test.copy()

# Apply same feature engineering
test_df = calculate_features(test_df)

# Use same price threshold from training data
test_df['is_expensive_basket'] = (test_df['total_basket_price'] > price_thresh).astype(int)

# Select same features as training
X_test = test_df[available_features].fillna(0)
X_test_scaled = scaler.transform(X_test)

print(f"Test data ready: {X_test.shape}")

# ===============================
# PREDICTION AND SUBMISSION
# ===============================

print("\nGenerating predictions...")
lgb_preds = models["LightGBM"].predict_proba(X_test)[:, 1]

# Create submission
id_column = 'ID' if 'ID' in test_df.columns else test_df.columns[0]

submission = pd.DataFrame({
    "ID": test_df[id_column],
    "fraud_flag": lgb_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(f"Number of predictions: {len(submission)}")
print(f"Predicted fraud rate: {lgb_preds.mean():.4%}")
print(submission.head(10))