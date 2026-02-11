import io
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

# Config: raw file URLs (change if needed)
TRAINING_URL = "https://raw.githubusercontent.com/abarramda/tempfiles/refs/heads/main/Demand%20MVR%20%26%20Nonlinear_lite%20NE.csv"
SCORING_URL = "https://raw.githubusercontent.com/abarramda/tempfiles/refs/heads/main/Regression%20test%20-%20Cleveland.csv"
OUTPUT_XLSX = "NE Model – Cleveland Scoring.xlsx"

EPS = 1e-6

INCOME_COLS = [
    "Income band A trips share",
    "Income band B trips share",
    "Income band C trips share",
    "Income band D trips share",
    "Income band E trips share",
]

OPTIONAL_TRIP_SHARE_COLS = [
    "Commute trips share",
    "Personal trips share",
    "Business trips share",
]

BASE_FEATURES = [
    "Beeline distance (mi)",
    "Road distance (mi)",
    "eVTOL time (min)",
    "Road time (min)",
    "eVTOL cost (USD)",
    "eVTOL cost per mile (USD/mi)",
] + INCOME_COLS

TARGET_COL = "Market penetration (%)"

@dataclass
class ModelResult:
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    r2: float
    aic: float
    bic: float

def _percent_to_frac(series: pd.Series) -> pd.Series:
    """
    Convert percent strings like '42%' to fraction 0.42.
    If already numeric or empty, handle gracefully.
    """
    def to_frac(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.strip()
            if x.endswith("%"):
                try:
                    return float(x[:-1]) / 100.0
                except ValueError:
                    return np.nan
            # allow '0.0%' already handled, or plain numeric string
            try:
                return float(x) / 100.0
            except ValueError:
                return np.nan
        # numeric like 0.42 or 42.0
        try:
            val = float(x)
            # heuristic: if >1 and <=100, treat as percent
            if val > 1 and val <= 100:
                return val / 100.0
            return val
        except Exception:
            return np.nan
    return series.apply(to_frac)

def _clean_and_engineer(df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
    # Normalize columns
    df = df.copy()

    # Target to fraction
    y_frac = _percent_to_frac(df.get(TARGET_COL))
    df["_penetration_frac"] = y_frac

    # Parse feature percentages to fractions
    for col in INCOME_COLS + OPTIONAL_TRIP_SHARE_COLS:
        if col in df.columns:
            df[col] = _percent_to_frac(df[col])

    # Engineer features
    # Distance: prefer Beeline; fallback to Road (row-wise)
    beeline = df.get("Beeline distance (mi)")
    road_dist = df.get("Road distance (mi)")
    df["_distance_mi"] = np.where(~pd.isna(beeline), beeline, road_dist)

    # Time advantage
    evtol_time = df.get("eVTOL time (min)")
    road_time = df.get("Road time (min)")
    if evtol_time is not None and road_time is not None:
        df["_time_adv_min"] = road_time - evtol_time
    else:
        df["_time_adv_min"] = np.nan

    # Cost features (as-is)
    # Optionally ratios can be added later
    # df["eVTOL cost (USD)"], df["eVTOL cost per mile (USD/mi)"]

    # Intersection rule for scoring:
    # Detect columns that are fully empty in the SCORING set and exclude them globally.
    # We'll check SCORING later; here we just prepare.

    return df

def _select_feature_columns(scoring_df: pd.DataFrame) -> List[str]:
    """
    Apply intersection rule: exclude any column that is fully empty in scoring.
    """
    cols = ["_distance_mi", "_time_adv_min", "eVTOL cost (USD)", "eVTOL cost per mile (USD/mi)"] + INCOME_COLS
    selected = []
    for c in cols:
        if c not in scoring_df.columns:
            continue
        # Fully empty means all NaN in scoring
        s = scoring_df[c]
        if s.notna().sum() == 0:
            # exclude
            continue
        selected.append(c)
    return selected

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))

def _inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def fit_model(train_df: pd.DataFrame, feature_cols: List[str]) -> ModelResult:
    # Keep rows with target and features present
    mask = ~train_df["_penetration_frac"].isna()
    for c in feature_cols:
        mask &= ~train_df[c].isna()
    df = train_df.loc[mask].copy()

    y = _logit(df["_penetration_frac"].values)
    X = df[feature_cols].astype(float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    return ModelResult(
        params=model.params,
        bse=model.bse,
        tvalues=model.tvalues,
        pvalues=model.pvalues,
        conf_int=model.conf_int(alpha=0.05),
        r2=float(model.rsquared),
        aic=float(model.aic),
        bic=float(model.bic),
    )

def cross_validate(train_df: pd.DataFrame, feature_cols: List[str], n_splits: int = 5, seed: int = 42):
    mask = ~train_df["_penetration_frac"].isna()
    for c in feature_cols:
        mask &= ~train_df[c].isna()
    df = train_df.loc[mask].copy()

    y_frac = df["_penetration_frac"].astype(float).values
    X_all = sm.add_constant(df[feature_cols].astype(float).values)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y_frac)

    for train_idx, test_idx in kf.split(X_all):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr = _logit(y_frac[train_idx])
        mdl = sm.OLS(y_tr, X_tr).fit(cov_type="HC3")
        z_hat = mdl.predict(X_te)
        preds[test_idx] = _inv_logit(z_hat)

    # Metrics on fraction scale
    resid = y_frac - preds
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    # R^2 on fraction scale
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_frac - np.mean(y_frac)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"rmse": rmse, "mae": mae, "r2": r2}

def score(scoring_df: pd.DataFrame, feature_cols: List[str], params: pd.Series) -> pd.DataFrame:
    df = scoring_df.copy()
    # Rows with at least one feature present
    mask = np.ones(len(df), dtype=bool)
    for c in feature_cols:
        mask &= ~df[c].isna()
    df_sc = df.loc[mask].copy()

    X = df_sc[feature_cols].astype(float)
    # Force add constant even if some features are constant (e.g., income bands)
    X = sm.add_constant(X, has_constant='add')
    z_hat = np.dot(X.values, params.values)
    p_hat = _inv_logit(z_hat)

    out = df_sc.copy()
    out["Predicted Market penetration (%)"] = (p_hat * 100.0)
    return out

def load_csv(url_or_path: str) -> pd.DataFrame:
    # pandas can read directly from raw GitHub
    return pd.read_csv(url_or_path)

def write_excel(
    training_df: pd.DataFrame,
    scoring_input_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    feature_cols: List[str],
    model_res: ModelResult,
    cv_metrics: dict,
    output_path: str,
):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # 1) Readme
        readme = pd.DataFrame({
            "Item": [
                "Training source",
                "Scoring source",
                "Run date",
                "Target transform",
                "Features used",
                "Excluded columns (scoring fully empty)",
            ],
            "Value": [
                TRAINING_URL,
                SCORING_URL,
                pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
                "logit(p) on penetration fraction",
                ", ".join(feature_cols),
                ", ".join([c for c in OPTIONAL_TRIP_SHARE_COLS if c not in feature_cols]),
            ],
        })
        readme.to_excel(writer, sheet_name="Readme", index=False)

        # 2) Training summary
        summary_cols = ["_penetration_frac", "_distance_mi", "_time_adv_min", "eVTOL cost (USD)", "eVTOL cost per mile (USD/mi)"] + INCOME_COLS
        desc_df = training_df[summary_cols].describe()
        desc_df.to_excel(writer, sheet_name="Training summary")
        # Correlation matrix - calculate offset dynamically
        corr = training_df[summary_cols].corr()
        start_row = len(desc_df) + 3  # Add buffer rows after describe()
        corr.to_excel(writer, sheet_name="Training summary", startrow=start_row)

        # 3) Model spec
        spec = pd.DataFrame({
            "Feature": ["const"] + feature_cols,
            "Coefficient": [model_res.params.get("const", np.nan)] + [model_res.params.get(c, np.nan) for c in feature_cols],
            "Robust SE (HC3)": [model_res.bse.get("const", np.nan)] + [model_res.bse.get(c, np.nan) for c in feature_cols],
            "t-stat": [model_res.tvalues.get("const", np.nan)] + [model_res.tvalues.get(c, np.nan) for c in feature_cols],
            "p-value": [model_res.pvalues.get("const", np.nan)] + [model_res.pvalues.get(c, np.nan) for c in feature_cols],
        })
        spec.to_excel(writer, sheet_name="Coefficients", index=False)

        # Confidence intervals
        ci = model_res.conf_int.copy()
        ci.columns = ["CI low (95%)", "CI high (95%)"]
        ci.to_excel(writer, sheet_name="Coefficients", startrow=len(spec) + 3)

        # 4) Diagnostics
        diag = pd.DataFrame({
            "Metric": ["R² (train, logit)", "AIC", "BIC", "CV R² (fraction)", "CV RMSE (fraction)", "CV MAE (fraction)"],
            "Value": [model_res.r2, model_res.aic, model_res.bic, cv_metrics["r2"], cv_metrics["rmse"], cv_metrics["mae"]],
        })
        diag.to_excel(writer, sheet_name="Diagnostics", index=False)

        # 5) Scoring – Cleveland
        keep_cols = [
            "Predicted Market penetration (%)",
            "_distance_mi", "_time_adv_min",
            "Beeline distance (mi)", "Road distance (mi)",
            "eVTOL time (min)", "Road time (min)",
            "eVTOL cost (USD)", "eVTOL cost per mile (USD/mi)",
        ] + INCOME_COLS
        cols_existing = [c for c in keep_cols if c in scored_df.columns]
        scored_final = scored_df[cols_existing].copy()
        # Dynamic percentage display: format shares and predicted penetration as %
        scored_final.to_excel(writer, sheet_name="Scoring – Cleveland", index=True)

        # 6) Charts
        workbook  = writer.book
        sheet_charts = workbook.add_worksheet("Charts")

        # Chart: Predicted vs distance
        # Write a small table for chart source
        chart_data = scored_final[["Predicted Market penetration (%)"]].copy()
        chart_data.insert(0, "distance_used (mi)", scored_df["_distance_mi"])
        chart_data.to_excel(writer, sheet_name="Charts", startrow=0, startcol=0, index=False)

        chart1 = workbook.add_chart({"type": "scatter"})
        chart1.add_series({
            "name": "Predicted vs distance",
            "categories": ["Charts", 1, 0, len(chart_data), 0],
            "values":     ["Charts", 1, 1, len(chart_data), 1],
            "marker": {"type": "circle", "size": 4},
        })
        chart1.set_title({"name": "Predicted penetration vs Distance"})
        chart1.set_x_axis({"name": "Distance (mi)"})
        chart1.set_y_axis({"name": "Predicted penetration (%)"})
        sheet_charts.insert_chart("E2", chart1)

        # Chart: Predicted vs time_advantage
        chart_data2 = scored_final[["Predicted Market penetration (%)"]].copy()
        chart_data2.insert(0, "time_advantage (min)", scored_df["_time_adv_min"])
        chart_data2.to_excel(writer, sheet_name="Charts", startrow=0, startcol=10, index=False)

        chart2 = workbook.add_chart({"type": "scatter"})
        chart2.add_series({
            "name": "Predicted vs time advantage",
            "categories": ["Charts", 1, 10, len(chart_data2), 10],
            "values":     ["Charts", 1, 11, len(chart_data2), 11],
            "marker": {"type": "circle", "size": 4},
        })
        chart2.set_title({"name": "Predicted penetration vs Time advantage"})
        chart2.set_x_axis({"name": "Time advantage (min)"})
        chart2.set_y_axis({"name": "Predicted penetration (%)"})
        sheet_charts.insert_chart("E20", chart2)

        # Chart: Coefficient magnitude
        coef_df = spec[["Feature", "Coefficient"]].copy()
        coef_df.to_excel(writer, sheet_name="Charts", startrow=30, startcol=0, index=False)
        chart3 = workbook.add_chart({"type": "column"})
        chart3.add_series({
            "name": "Coefficients",
            "categories": ["Charts", 31, 0, 31 + len(coef_df) - 1, 0],
            "values":     ["Charts", 31, 1, 31 + len(coef_df) - 1, 1],
        })
        chart3.set_title({"name": "Coefficient magnitudes (logit model)"})
        chart3.set_x_axis({"name": "Features"})
        chart3.set_y_axis({"name": "Coefficient"})
        sheet_charts.insert_chart("E38", chart3)

def main():
    # Load data
    train_raw = load_csv(TRAINING_URL)
    scoring_raw = load_csv(SCORING_URL)

    # Clean/feature engineer
    train_df = _clean_and_engineer(train_raw, is_training=True)
    scoring_df = _clean_and_engineer(scoring_raw, is_training=False)

    # Intersection rule based on scoring availability
    feature_cols = _select_feature_columns(scoring_df)

    # Fit
    model_res = fit_model(train_df, feature_cols)

    # CV
    cv_metrics = cross_validate(train_df, feature_cols, n_splits=5, seed=42)

    # Score
    scored_df = score(scoring_df, feature_cols, model_res.params)

    # Write Excel
    write_excel(train_df, scoring_df, scored_df, feature_cols, model_res, cv_metrics, OUTPUT_XLSX)

if __name__ == "__main__":
    main()
