"""
Statistical and Predictive Modeling Module for Insurance Risk-Based Pricing.

Provides reusable pipelines for preparing data, training regressors
(Claim Severity) and classifiers (Claim Probability), and applying
the premium pricing framework.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------------------------------
# Data Preprocessing & Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.
    """
    out = df.copy()
    
    # 1. Vehicle Age (current year 2026 minus registration year)
    if "RegistrationYear" in out.columns:
        # Some values could be unreasonable (e.g. 9999). Bound them sensibly.
        reg_year = pd.to_numeric(out["RegistrationYear"], errors="coerce")
        out["VehicleAge"] = 2026 - reg_year
        out["VehicleAge"] = out["VehicleAge"].clip(0, 100)
    else:
        out["VehicleAge"] = 10  # default median
        
    # 2. Premium to Sum Insured Ratio (Proxy for baseline pricing risk density)
    if "TotalPremium" in out.columns and "SumInsured" in out.columns:
        out["PremiumToSumInsured"] = out["TotalPremium"] / (out["SumInsured"] + 1.0)
    else:
        out["PremiumToSumInsured"] = 0.0

    return out


def prepare_modeling_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    is_regression: bool = True,
    test_size: float = 0.3,
    random_state: int = 42,
    encoder: OrdinalEncoder | None = None,
    cat_imputer: SimpleImputer | None = None,
    num_imputer: SimpleImputer | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], OrdinalEncoder | None, SimpleImputer | None, SimpleImputer | None]:
    """
    Preprocess data for training/testing. Handles imputation, categorical
    encoding, and train-test splitting.
    """
    # Feature engineering
    data = engineer_features(df)
    
    # Select features & target
    X = data[feature_cols].copy()
    y = data[target_col].copy()

    # Identify numeric and categorical columns
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # Impute categorical features
    if len(cat_cols) > 0:
        if cat_imputer is None:
            cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols].astype(str))
        else:
            X[cat_cols] = cat_imputer.transform(X[cat_cols].astype(str))
            
        if encoder is None:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[cat_cols] = encoder.fit_transform(X[cat_cols])
        else:
            X[cat_cols] = encoder.transform(X[cat_cols])
    else:
        encoder = None
        cat_imputer = None

    # Impute numerical features
    if len(num_cols) > 0:
        if num_imputer is None:
            num_imputer = SimpleImputer(strategy="median")
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
        else:
            X[num_cols] = num_imputer.transform(X[num_cols])
    else:
        num_imputer = None

    # Train/test split
    # Handle stratification safely in case of tiny class sizes in test dummy data
    stratify_y = None if is_regression else y
    if stratify_y is not None:
        if len(stratify_y) < 2 or stratify_y.nunique() < 2 or (stratify_y.value_counts() < 2).any():
            stratify_y = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    return (
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values,
        feature_cols,
        encoder,
        cat_imputer,
        num_imputer,
    )


# ---------------------------------------------------------------------------
# Training Pipelines
# ---------------------------------------------------------------------------

def train_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train Linear Regression, Random Forest Regressor, and XGBoost Regressor.
    """
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # 3. XGBoost Regressor
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    return {
        "Linear Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb,
    }


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train Logistic Regression, Random Forest Classifier, and XGBoost Classifier.
    """
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train, y_train)

    # 2. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    # 3. XGBoost Classifier
    # Compute class weights for scale_pos_weight to handle remaining imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos = (neg_count / pos_count) if pos_count > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    return {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb,
    }


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def evaluate_regressors(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Compute RMSE and R2 for regression models.
    """
    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "RMSE": rmse, "R2": r2})
    return pd.DataFrame(results)


def evaluate_classifiers(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Compute Accuracy, Precision, Recall, and F1 for classification models.
    """
    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Pricing Framework
# ---------------------------------------------------------------------------

def calibrate_probability(
    p_model: np.ndarray,
    population_rate: float,
    sample_rate: float,
    is_weighted: bool = True,
) -> np.ndarray:
    """
    Calibrate model probabilities back to the population rate.
    If the model was trained with class weights or scale_pos_weight (to maximize F1/Recall),
    set is_weighted=True, which cancels out the downsampling and weighting down to:
        p = (p_m * r) / (p_m * r + (1 - p_m) * (1 - r))
    If the model was trained on the downsampled dataset WITHOUT class weights,
    set is_weighted=False, which uses Platt scaling / prior adjustment:
        p = p_m / (beta + (1 - beta) * p_m)
        where beta = ((1 - r) / r) * (s / (1 - s))
    """
    r = population_rate
    s = sample_rate
    
    if is_weighted:
        return (p_model * r) / (p_model * r + (1.0 - p_model) * (1.0 - r))
    else:
        if r <= 0 or r >= 1 or s <= 0 or s >= 1:
            return p_model
        beta = ((1.0 - r) / r) * (s / (1.0 - s))
        return p_model / (beta + (1.0 - beta) * p_model)


def calculate_risk_based_premium(
    p_claim: np.ndarray,
    predicted_severity: np.ndarray,
    expense_loading: float = 0.20,
    profit_margin: float = 0.10,
) -> np.ndarray:
    """
    Apply risk-based pricing framework formula:
    Premium = (P(claim) * Predicted Severity) * (1 + expense_loading + profit_margin)
    """
    expected_claim_cost = p_claim * predicted_severity
    loading_multiplier = 1.0 + expense_loading + profit_margin
    return expected_claim_cost * loading_multiplier
