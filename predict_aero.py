# predict_aero.py
# Batch predictions for Cl, Cd, Cdp, Cm from an Excel file of new cases.
# Uses artifacts saved by train_lstm_aero.py (model, scalers, encoder, meta).

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ------------------------
# Default filenames
# ------------------------
DEFAULT_INPUT = "Testing Dataset - NACA 4 Digit Airfoils.xlsx"
OUT_DIR = Path(".")
MODELS_DIR = Path("models")

DEFAULT_CSV  = "predictions.csv"
DEFAULT_XLSX = "predictions.xlsx"

REQUIRED_COLS = ["Airfoil", "Re", "Alpha", "Top_Xtr", "Bot_Xtr"]

def load_artifacts(models_dir: Path):
    meta_path = models_dir / "meta.json"
    model_path = models_dir / "aero_lstm.keras"
    x_scaler_path = models_dir / "x_scaler.pkl"
    y_scaler_path = models_dir / "y_scaler.pkl"
    ohe_path = models_dir / "airfoil_encoder.pkl"

    if not meta_path.exists():
        sys.exit(f"[ERROR] Missing {meta_path}. Train the model first.")
    if not model_path.exists():
        sys.exit(f"[ERROR] Missing {model_path}. Train the model first.")
    if not x_scaler_path.exists():
        sys.exit(f"[ERROR] Missing {x_scaler_path}.")
    if not y_scaler_path.exists():
        sys.exit(f"[ERROR] Missing {y_scaler_path}.")
    if not ohe_path.exists():
        sys.exit(f"[ERROR] Missing {ohe_path}.")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    model = keras.models.load_model(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    ohe = joblib.load(ohe_path)

    model_type = meta.get("model_type", "mlp")
    num_features = meta.get("num_features", ["Re", "Alpha", "Top_Xtr", "Bot_Xtr"])
    targets = meta.get("targets", ["Cl", "Cd", "Cdp", "Cm"])

    return model, x_scaler, y_scaler, ohe, model_type, num_features, targets

def validate_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Input file is missing columns: {missing}\n"
                 f"Expected at least: {required_cols}")

def preprocess(df: pd.DataFrame, num_features, ohe, x_scaler, model_type):
    # Select numeric features and coerce
    X_num = df[num_features].apply(pd.to_numeric, errors="coerce")
    if X_num.isna().any().any():
        bad_rows = X_num[X_num.isna().any(axis=1)]
        raise ValueError(f"Found non-numeric values in numeric features for rows:\n{bad_rows}")

    # One-hot encode Airfoil (unknown labels -> ignored, gives all zeros for that row)
    airfoil = df["Airfoil"].astype(str).values.reshape(-1, 1)
    X_cat = ohe.transform(airfoil)  # handle_unknown='ignore' set during training

    # Scale numeric features
    X_num_sc = x_scaler.transform(X_num.values.astype(np.float32))

    # Concatenate numeric + categorical
    X_full = np.hstack([X_num_sc, X_cat]).astype(np.float32)

    # Reshape for LSTM if needed
    if model_type == "lstm":
        X_in = X_full[:, None, :]
    else:
        X_in = X_full
    return X_in

def main():
    parser = argparse.ArgumentParser(description="Batch predict aero coefficients from Excel.")
    parser.add_argument("--input",  default=DEFAULT_INPUT, help="Path to input Excel file")
    parser.add_argument("--out_csv",  default=DEFAULT_CSV,  help="Output CSV filename")
    parser.add_argument("--out_xlsx", default=DEFAULT_XLSX, help="Output Excel filename")
    args = parser.parse_args()

    # Load artifacts
    model, x_scaler, y_scaler, ohe, model_type, num_features, targets = load_artifacts(MODELS_DIR)

    print("[INFO] Loaded artifacts.")
    print("       model_type   :", model_type)
    print("       num_features :", num_features)
    print("       targets      :", targets)

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[ERROR] Input file not found: {in_path}")

    # Read Excel
    print(f"[INFO] Reading: {in_path}")
    df = pd.read_excel(in_path)

    # Basic validation
    validate_columns(df, REQUIRED_COLS)

    # Keep only the columns we actually need; preserve original order for output
    df_in = df[["Airfoil"] + num_features].copy()

    # Preprocess
    try:
        X_in = preprocess(df_in, num_features, ohe, x_scaler, model_type)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    # Predict (scaled → inverse transform to original units)
    print("[INFO] Predicting...")
    y_pred_sc = model.predict(X_in, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_sc)

    # Prepare output frame
    pred_df = pd.DataFrame(y_pred, columns=targets)
    out_df = pd.concat([df_in.reset_index(drop=True), pred_df], axis=1)

    # Save CSV + Excel
    out_csv  = OUT_DIR / args.out_csv
    out_xlsx = OUT_DIR / args.out_xlsx

    out_df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Predictions")

    print(f"[OK] Wrote: {out_csv.resolve()}")
    print(f"[OK] Wrote: {out_xlsx.resolve()}")

if __name__ == "__main__":
    # Quiet TF’s oneDNN messages if desired:
    # import os; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    main()
