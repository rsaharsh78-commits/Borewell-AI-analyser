"""
train_model.py - Train XGBoost models on real Hyderabad data for JalDrishti v2.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hyderabad_training_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Feature configuration
NUMERIC_FEATURES = [
    'rainfall_15yr_annual_avg_mm', 'rainfall_monsoon_avg_mm', 'et0_annual_avg_mm',
    'effective_recharge_mm', 'heavy_rain_days_per_yr', 'moderate_rain_days_per_yr',
    'rainfall_intensity_ratio', 'drought_years_in_15',
    'elevation_m', 'slope_degrees', 'twi', 'distance_to_waterbody_m',
    'impervious_surface_pct',
    'gwl_pre_monsoon_mbgl', 'gwl_post_monsoon_mbgl', 'gwl_seasonal_fluctuation_m',
    'extraction_stage_pct', 'weathered_zone_thickness_m', 'depth_to_bedrock_m',
    'lineament_density_per_km2',
    'clay_pct', 'sand_pct', 'soil_ph',
    'planned_depth_ft', 'planned_month',
]

CATEGORICAL_FEATURES = ['rock_type', 'aquifer_type', 'land_cover_class']
TARGET_CLASS = 'borewell_success'
TARGET_DEPTH = 'avg_borewell_depth_ft'


def main():
    print("[OK] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Dataset shape: {df.shape}")
    print(f"[OK] Success rate: {df[TARGET_CLASS].mean():.3f}")

    # Fill missing values
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"[OK] Encoded {col}: {list(le.classes_)}")

    # Build feature list
    feature_cols = []
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            feature_cols.append(col)
    for col in CATEGORICAL_FEATURES:
        enc_col = col + '_encoded'
        if enc_col in df.columns:
            feature_cols.append(enc_col)

    print(f"[OK] Using {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols].copy()
    X = X.fillna(0)

    # ---- Risk Classifier ----
    print("\n[OK] Training Risk Classifier...")
    y_class = df[TARGET_CLASS]
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    print(f"[OK] Classification -> AUC: {auc:.4f}, F1: {f1:.4f}")

    # ---- Depth Regressor ----
    print("\n[OK] Training Depth Regressor...")
    if TARGET_DEPTH in df.columns:
        y_depth = df[TARGET_DEPTH]
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_depth, test_size=0.2, random_state=42)

        reg = XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        reg.fit(X_train_d, y_train_d)

        y_pred_d = reg.predict(X_test_d)
        mae = mean_absolute_error(y_test_d, y_pred_d)
        print(f"[OK] Regression -> MAE: {mae:.2f} ft")
    else:
        reg = None
        print("[WARN] No depth target column found, skipping depth regressor")

    # ---- SHAP ----
    print("\n[OK] Generating SHAP summary plot...")
    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] SHAP plot saved to models/shap_summary.png")
    except Exception as e:
        print(f"[WARN] SHAP plot generation failed: {e}")

    # ---- Feature Importance ----
    print("\n[OK] Top 10 Feature Importances:")
    importances = clf.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for name, imp in feat_imp[:10]:
        bar = '#' * int(imp * 100)
        print(f"  {name:35s} {imp:.4f} {bar}")

    # ---- Save Models ----
    print("\n[OK] Saving models and artifacts...")
    joblib.dump(clf, MODELS_DIR / "risk_model.joblib")
    joblib.dump(reg, MODELS_DIR / "depth_model.joblib") if reg else None

    config = {
        'features': feature_cols,
        'numeric_features': [c for c in feature_cols if c in NUMERIC_FEATURES],
        'categorical_features': CATEGORICAL_FEATURES,
        'encoders': encoders,
        'model_version': '2.0-hyderabad',
        'auc': auc,
        'f1': f1,
    }
    joblib.dump(config, MODELS_DIR / "feature_config.joblib")

    print(f"\n[OK] Pipeline complete! Models saved to {MODELS_DIR}")
    print(f"[OK] Risk model: AUC={auc:.4f}, F1={f1:.4f}")
    if reg:
        print(f"[OK] Depth model: MAE={mae:.2f} ft")


if __name__ == "__main__":
    main()
