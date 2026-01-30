# ============================================================
# train_mlp_aero.py  (MLP-only, weighted loss, clean prints)
# ------------------------------------------------------------
# This script trains a neural network (an MLP = Multi-Layer Perceptron)
# to predict four aerodynamic coefficients:
#   Cl, Cd, Cdp, Cm
# from inputs:
#   Re (Reynolds number),
#   Alpha (angle of attack),
#   Top_Xtr (top surface transition location),
#   Bot_Xtr (bottom surface transition location),
#   and Airfoil ID (encoded as one-hot).
#
# The script:
#   1. Loads data from Excel.
#   2. Cleans and splits into train / validation / test.
#   3. Scales features and encodes airfoil IDs.
#   4. Builds and trains an MLP with a weighted loss (so Cd/Cdp matter slightly more).
#   5. Evaluates on test data, prints metrics, and saves plots and model files.
# ============================================================

import os                  # Provides functions to work with files and directories.
import json                # Allows saving/loading data in JSON format (for meta info).
import argparse            # Handles command-line arguments (so you can change settings without editing code).
import joblib              # Used to save and load Python objects (like scalers and encoders) to disk.
import numpy as np         # Fundamental library for numerical computations and arrays.
import pandas as pd        # Library for working with tabular data (like Excel spreadsheets).
import matplotlib.pyplot as plt  # Library for plotting graphs and figures.

# Import specific tools from scikit-learn:
from sklearn.model_selection import train_test_split   # Function to split data into train/val/test sets.
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # Tools to scale numbers and encode categories.
from sklearn.metrics import r2_score                   # Function to compute R² (coefficient of determination).

# Import TensorFlow and Keras, the deep learning framework we're using:
import tensorflow as tf
from tensorflow import keras

# For convenience, create short aliases to some Keras modules:
layers     = keras.layers       # Access to different layer types (Dense, Input, etc.).
models     = keras.models       # Used to create and manage models.
optimizers = keras.optimizers   # Optimizers (like Adam) used to update weights.
callbacks  = keras.callbacks    # Callbacks that can stop training early, adjust learning rate, etc.

# --------------------------
# Config (defaults you can override via CLI)
# --------------------------

# Name of the Excel file containing the training data.
DATA_FILE   = "Complete Training Dataset (All NACA Airfoils).xlsx"

# List of numeric input features we will use from the dataset.
NUM_FEATURES = ["Re", "Alpha", "Top_Xtr", "Bot_Xtr"]

# List of output targets (the coefficients we want to predict).
TARGETS      = ["Cl", "Cd", "Cdp", "Cm"]

# Fractions of the dataset to allocate to test and validation sets.
TEST_SIZE = 0.15   # 15% of data reserved for testing (final evaluation).
VAL_SIZE  = 0.15   # 15% of data reserved for validation (used during training).

# Seed for randomness so that results are reproducible (same splits, same initial weights).
RANDOM_SEED = 42

# Training hyperparameters:
EPOCHS        = 300     # Maximum number of passes through the training data.
BATCH_SIZE    = 64      # Number of samples processed at once before updating weights.
LEARNING_RATE = 1e-3    # Step size for the optimizer when updating weights.

# Directory where models, scalers, plots, and metrics will be saved.
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)  # Create the directory if it doesn't exist.

# --------------------------
# Helpers
# --------------------------

def set_reproducible(seed=42):
    """
    Purpose:
        Make the training process reproducible. Neural networks use randomness
        (for example, in initializing weights and shuffling data). Setting seeds
        ensures you get the same result each time you run the script.
    How it works:
        - np.random.seed: controls NumPy's random number generator.
        - tf.random.set_seed: controls TensorFlow's random number generator.
    """
    np.random.seed(seed)         # Fix NumPy's random seed.
    tf.random.set_seed(seed)     # Fix TensorFlow's random seed.


def coerce_numeric(df, cols):
    """
    Purpose:
        Ensure that specific columns in a pandas DataFrame are numeric.
        If there are invalid values (like text in a numeric column),
        they are turned into NaN (Not-a-Number), which we later drop.
    Parameters:
        df   : the DataFrame (table of data).
        cols : list of column names we want to convert to numeric.
    Returns:
        The DataFrame with columns converted to numeric type.
    """
    for c in cols:  # Loop over each column name in cols.
        # Convert the column to numeric; errors='coerce' turns bad values into NaN.
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df       # Return the modified DataFrame.


def metrics_report(y_true, y_pred, names):
    """
    Purpose:
        Compute and print error metrics comparing true vs predicted values on the test set.
        Also return these metrics in a dictionary for saving to disk.
    Parameters:
        y_true : array of true (actual) values of the targets.
        y_pred : array of predicted values from the model.
        names  : list of target names (["Cl", "Cd", "Cdp", "Cm"]).
    Metrics:
        MAE  (Mean Absolute Error) - average absolute difference.
        RMSE (Root Mean Squared Error) - penalizes large errors more.
        R²   - how much variance in the data is explained by the model (1.0 is perfect).
    """

    # Compute MAE for each target (column).
    mae_each  = np.mean(np.abs(y_true - y_pred), axis=0)

    # Compute RMSE for each target.
    rmse_each = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    # Compute R² for each target independently.
    r2_each   = r2_score(y_true, y_pred, multioutput="raw_values")

    # Compute an overall R², weighted by variance of each target.
    r2_over   = r2_score(y_true, y_pred, multioutput="variance_weighted")

    # Print nicely formatted metrics to the console.
    print("\n=== Test Metrics (original units) ===")
    for i, name in enumerate(names):  # Loop over each target name and index.
        print(f"{name:>4}  MAE = {mae_each[i]:.6f}   RMSE = {rmse_each[i]:.6f}   R² = {r2_each[i]:.4f}")
    print(f"Overall MAE  = {np.mean(mae_each):.6f}")
    print(f"Overall RMSE = {np.mean(rmse_each):.6f}")
    print(f"Overall R²   = {r2_over:.4f}\n")

    # Return the metrics in a dictionary for saving or further processing.
    return {
        "mae_per_target":  {n: float(mae_each[i])  for i, n in enumerate(names)},
        "rmse_per_target": {n: float(rmse_each[i]) for i, n in enumerate(names)},
        "r2_per_target":   {n: float(r2_each[i])   for i, n in enumerate(names)},
        "overall_mae":  float(np.mean(mae_each)),
        "overall_rmse": float(np.mean(rmse_each)),
        "overall_r2":   float(r2_over),
    }


def plot_history(history, out_png):
    """
    Purpose:
        Plot the training and validation loss curves over epochs.
        This helps you see if the model is learning, overfitting, or underfitting.
    Parameters:
        history : object returned by model.fit(), containing loss values for each epoch.
        out_png : filename (path) where the plot image will be saved.
    """
    plt.figure()  # Start a new figure.

    # Plot training loss (MSE) against epochs.
    plt.plot(history.history["loss"], label="train_loss")

    # If validation loss is present, plot it as well.
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")

    plt.xlabel("Epoch")          # Label x-axis.
    plt.ylabel("MSE loss")       # Label y-axis.
    plt.title("Training History")  # Plot title.
    plt.legend()                 # Show legend (train vs val).
    plt.tight_layout()           # Adjust layout so everything fits.
    plt.savefig(out_png, dpi=150)  # Save figure to the specified file.
    plt.close()                  # Close the figure to free memory.


def plot_pred_vs_true(y_true, y_pred, names, out_dir):
    """
    Purpose:
        Create scatter plots comparing predicted vs true values for each coefficient.
        Ideal predictions lie on the y=x (diagonal) line. Deviations show error.
    Parameters:
        y_true : true values (test set).
        y_pred : predicted values (from the model).
        names  : list of coefficient names.
        out_dir: directory in which to save the plots.
    """

    # First, create one plot per target coefficient.
    for i, name in enumerate(names):
        plt.figure()
        plt.scatter(y_true[:, i], y_pred[:, i], s=12)  # Each point = one test sample.
        # Compute min and max across true and pred to define a square plotting range.
        mn = float(min(y_true[:, i].min(), y_pred[:, i].min()))
        mx = float(max(y_true[:, i].max(), y_pred[:, i].max()))
        plt.plot([mn, mx], [mn, mx])  # Draw the ideal 1:1 line.
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"Predicted vs True: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pred_vs_true_{name}.png"), dpi=150)
        plt.close()

    # Now, create a combined 2x2 grid of all four plots.
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()  # Flatten 2D array of axes into 1D list.

    for i, name in enumerate(names):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], s=12)
        mn = float(min(y_true[:, i].min(), y_pred[:, i].min()))
        mx = float(max(y_true[:, i].max(), y_pred[:, i].max()))
        ax.plot([mn, mx], [mn, mx])
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Pred {name}")
        ax.set_title(name)

    plt.suptitle("Predicted vs True (Test Set)")  # Overall title.
    plt.tight_layout(rect=[0,0.03,1,0.95])        # Adjust layout to make room for title.
    plt.savefig(os.path.join(out_dir, "pred_vs_true_grid.png"), dpi=150)
    plt.close()


def build_mlp(n_in, n_out, widths, lr=1e-3):
    """
    Purpose:
        Build a Multi-Layer Perceptron (MLP) model with adjustable hidden layer sizes.
    Parameters:
        n_in  : number of input features (numeric + airfoil one-hot).
        n_out : number of outputs (4 coefficients).
        widths: list of integers, e.g. [128, 128, 64]
                Each number is the number of neurons in a hidden layer.
        lr    : learning rate for the Adam optimizer.
    Returns:
        A compiled Keras model ready to be trained.
    """

    # Start a list of layers with the input layer specifying the input shape.
    layers_list = [layers.Input(shape=(n_in,))]

    # Add a Dense (fully-connected) hidden layer for each width value.
    for w in widths:
        layers_list.append(layers.Dense(w, activation="relu"))

    # Add the final output layer with n_out neurons and linear activation.
    layers_list.append(layers.Dense(n_out, activation="linear"))

    # Wrap all layers into a Sequential model.
    model = models.Sequential(layers_list)

    # Compile the model with Adam optimizer, MSE loss, and MAE metric.
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def weighted_mse_factory(weights_np):
    """
    Purpose:
        Create a custom loss function that applies different weights to each output's error.
        This lets us say, for example, "Cd and Cdp are more important than Cl and Cm."
    Parameters:
        weights_np : a NumPy array of shape (4,), e.g. [1.0, 1.5, 1.5, 1.0]
                     corresponding to [Cl_weight, Cd_weight, Cdp_weight, Cm_weight]
    Returns:
        A function loss_fn(y_true, y_pred) that Keras can use as the loss function.
    """
    w = tf.constant(weights_np, dtype=tf.float32)  # Convert weights to a TensorFlow constant.

    def loss_fn(y_true, y_pred):
        """
        Inner function that actually computes the weighted MSE.
        y_true, y_pred have shape (batch_size, 4).
        """
        err2 = tf.square(y_true - y_pred)   # Squared error for each target: (batch, 4)
        w_err2 = err2 * w                  # Multiply each column by its weight.
        # First average across the 4 outputs, then across the batch.
        return tf.reduce_mean(tf.reduce_mean(w_err2, axis=-1))
    return loss_fn  # Return the custom loss function to be used in model.compile.


def assess_accuracy(metrics_dict, thresholds):
    """
    Purpose:
        Decide whether the model is "good enough" according to user-provided thresholds.
        For example, require overall R² >= 0.97 and overall MAE <= 0.035.
    Parameters:
        metrics_dict: dictionary returned by metrics_report (contains overall and per-target metrics).
        thresholds  : dictionary defining required minimum R² and maximum MAE.
    Returns:
        A dictionary:
          {
            "adequate": True/False,
            "fail_reasons": [list of human-readable strings],
            "thresholds": thresholds
          }
    """

    ok = True          # Assume model is adequate until a check fails.
    msg = []           # List to store explanations for any failures.

    # Check overall R² (goodness of fit).
    if metrics_dict["overall_r2"] < thresholds["min_r2_overall"]:
        ok = False
        msg.append(f"- overall R² {metrics_dict['overall_r2']:.4f} < {thresholds['min_r2_overall']:.4f}")

    # Check overall MAE (average absolute error).
    if metrics_dict["overall_mae"] > thresholds["max_mae_overall"]:
        ok = False
        msg.append(f"- overall MAE {metrics_dict['overall_mae']:.6f} > {thresholds['max_mae_overall']:.6f}")

    # Optionally, check minimum R² per target (Cl, Cd, Cdp, Cm).
    for t, min_r2 in thresholds.get("per_target_min_r2", {}).items():
        r2t = metrics_dict["r2_per_target"].get(t, None)
        if r2t is not None and r2t < min_r2:
            ok = False
            msg.append(f"- {t} R² {r2t:.4f} < {min_r2:.4f}")

    # Return verdict and reasons.
    return {"adequate": ok, "fail_reasons": msg, "thresholds": thresholds}


def to_py_floats(d, decimals=6):
    """
    Purpose:
        Convert values in a dictionary from NumPy data types to plain Python floats,
        and round them to a specified number of decimal places.
        This makes printed output cleaner (no 'np.float32' displayed).
    Parameters:
        d        : dictionary with numeric values.
        decimals : how many decimal places to keep.
    Returns:
        New dictionary with clean Python floats.
    """
    return {k: round(float(v), decimals) for k, v in d.items()}

# --------------------------
# Main
# --------------------------

def main():
    """
    Purpose:
        The main function that:
          - Parses command-line arguments.
          - Loads and prepares the data.
          - Builds and trains the MLP model.
          - Evaluates on the test set.
          - Saves the model, scalers, encoder, metrics, and plots.
    """

    # Set up argument parser so you can adjust settings from the command line.
    ap = argparse.ArgumentParser(description="Train MLP aero model with weighted loss.")

    # Argument for hidden layer sizes (e.g. "128,128,64").
    ap.add_argument("--widths", type=str, default="128,128,64",
                    help="Comma-separated hidden layer sizes, e.g., 160,128,64")

    # Loss weights (default gives Cd/Cdp a little extra priority).
    ap.add_argument("--w_cl",  type=float, default=1.0)
    ap.add_argument("--w_cd",  type=float, default=1.5)
    ap.add_argument("--w_cdp", type=float, default=1.5)
    ap.add_argument("--w_cm",  type=float, default=1.0)

    # Adequacy thresholds for model quality.
    ap.add_argument("--min_r2_overall", type=float, default=0.97)
    ap.add_argument("--max_mae_overall", type=float, default=0.035)
    ap.add_argument("--min_r2_cl",  type=float, default=0.98)
    ap.add_argument("--min_r2_cd",  type=float, default=0.70)
    ap.add_argument("--min_r2_cdp", type=float, default=0.70)
    ap.add_argument("--min_r2_cm",  type=float, default=0.60)

    # Parse the arguments from the command line into an object called args.
    args = ap.parse_args()

    # Convert the widths string (like "128,128,64") into a Python list [128, 128, 64].
    widths = [int(s) for s in args.widths.split(",")]

    # Pack the loss weights into a NumPy array for [Cl, Cd, Cdp, Cm].
    loss_weights = np.array([args.w_cl, args.w_cd, args.w_cdp, args.w_cm], dtype=np.float32)

    # Collect thresholds into a dictionary for easier passing to assess_accuracy().
    thresholds = {
        "min_r2_overall": args.min_r2_overall,
        "max_mae_overall": args.max_mae_overall,
        "per_target_min_r2": {
            "Cl":  args.min_r2_cl,
            "Cd":  args.min_r2_cd,
            "Cdp": args.min_r2_cdp,
            "Cm":  args.min_r2_cm,
        },
    }

    # Set random seeds so runs are repeatable.
    set_reproducible(RANDOM_SEED)

    # 1) Load the data from Excel.
    print(f"Loading: {DATA_FILE}")
    df = pd.read_excel(DATA_FILE)

    # Build a list of all required columns: Airfoil ID, input features, and targets.
    required = ["Airfoil"] + NUM_FEATURES + TARGETS

    # Check which required columns are missing in the file, if any.
    missing = [c for c in required if c not in df.columns]
    if missing:
        # If any columns are missing, stop and show an error.
        raise ValueError(f"Missing expected columns in Excel: {missing}")

    # 2) Clean the data.
    df = df[required].copy()               # Keep only the required columns.
    df["Airfoil"] = df["Airfoil"].astype(str)  # Ensure Airfoil column is of string type.
    # Convert numeric columns and drop any rows with NaNs (invalid data).
    df = coerce_numeric(df, NUM_FEATURES + TARGETS).dropna(subset=NUM_FEATURES + TARGETS + ["Airfoil"])
    print("Data shape after cleaning:", df.shape)

    # 3) Convert DataFrame columns to NumPy arrays for model input.
    X_num_all   = df[NUM_FEATURES].values.astype(np.float32)    # Numeric inputs (Re, Alpha, etc.)
    airfoil_all = df["Airfoil"].values.reshape(-1, 1)           # Airfoil IDs as a 2D array (N x 1)
    Y_all       = df[TARGETS].values.astype(np.float32)        # Target coefficients

    # 4) Split into train/validation/test (randomly, but reproducibly due to RANDOM_SEED).
    X_num_tmp, X_num_test, airfoil_tmp, airfoil_test, y_tmp, y_test = train_test_split(
        X_num_all, airfoil_all, Y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Compute what fraction of the remaining data should go to validation.
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)

    # Now split the temporary set into train and validation sets.
    X_num_train, X_num_val, airfoil_train, airfoil_val, y_train, y_val = train_test_split(
        X_num_tmp, airfoil_tmp, y_tmp, test_size=val_ratio, random_state=RANDOM_SEED
    )

    print(f"Train: {X_num_train.shape}, Val: {X_num_val.shape}, Test: {X_num_test.shape}")

    # 5) One-hot encode Airfoil (convert categorical IDs into binary vectors).
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_train = ohe.fit_transform(airfoil_train)  # Learn categories from training set.
    X_cat_val   = ohe.transform(airfoil_val)       # Apply encoding to validation set.
    X_cat_test  = ohe.transform(airfoil_test)      # Apply encoding to test set.

    # 6) Scale numeric inputs to [0,1] range using MinMaxScaler.
    x_scaler = MinMaxScaler()
    X_num_train_sc = x_scaler.fit_transform(X_num_train)  # Fit scaler on training numeric data.
    X_num_val_sc   = x_scaler.transform(X_num_val)        # Apply same scaling to validation.
    X_num_test_sc  = x_scaler.transform(X_num_test)       # Apply same scaling to test.

    # Combine scaled numeric features and one-hot airfoil features into one input array.
    X_train_sc = np.hstack([X_num_train_sc, X_cat_train]).astype(np.float32)
    X_val_sc   = np.hstack([X_num_val_sc,   X_cat_val  ]).astype(np.float32)
    X_test_sc  = np.hstack([X_num_test_sc,  X_cat_test ]).astype(np.float32)

    # 7) Scale targets (Cl, Cd, Cdp, Cm) also to [0,1] for stable training.
    y_scaler = MinMaxScaler()
    y_train_sc = y_scaler.fit_transform(y_train)  # Fit scaler on training target values.
    y_val_sc   = y_scaler.transform(y_val)        # Apply to validation targets.
    y_test_sc  = y_scaler.transform(y_test)       # Apply to test targets.

    # 8) Build MLP model with weighted loss.
    n_in  = X_train_sc.shape[1]        # Total number of input features.
    n_out = len(TARGETS)               # Number of outputs (4 coefficients).
    model = build_mlp(n_in, n_out, widths, lr=LEARNING_RATE)  # Create MLP.

    # Compile the model again, this time using the weighted loss function.
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=weighted_mse_factory(loss_weights),   # Use our custom weighted MSE loss.
        metrics=["mae"]                            # Track MAE during training.
    )
    model.summary()  # Print a summary of the model architecture.

    # 9) Train the model using callbacks for early stopping and learning rate reduction.
    cb = [
        # Stop training if validation loss doesn't improve for 20 epochs.
        callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        # Reduce learning rate by factor 0.5 if val_loss plateaus for 8 epochs.
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=1),
        # Save the best model (lowest validation loss) to disk.
        callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "best.keras"),
                                  monitor="val_loss", save_best_only=True)
    ]

    # Actually train the model on the training data and validate on the validation data.
    history = model.fit(
        X_train_sc, y_train_sc,
        validation_data=(X_val_sc, y_val_sc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=cb
    )

    # 10) Save the trained model and preprocessing objects to disk.
    final_model_path = os.path.join(OUT_DIR, "aero_mlp.keras")
    model.save(final_model_path)  # Save the neural network weights and architecture.
    joblib.dump(x_scaler, os.path.join(OUT_DIR, "x_scaler.pkl"))   # Save input scaler.
    joblib.dump(y_scaler, os.path.join(OUT_DIR, "y_scaler.pkl"))   # Save target scaler.
    joblib.dump(ohe,      os.path.join(OUT_DIR, "airfoil_encoder.pkl"))  # Save airfoil encoder.

    # Save meta info (features, targets, widths, loss weights) as JSON.
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({
            "num_features": NUM_FEATURES,
            "targets": TARGETS,
            "model_type": "mlp",
            "airfoil_ohe_categories": [c.tolist() for c in ohe.categories_],
            "widths": widths,
            "loss_weights": [float(x) for x in loss_weights]
        }, f, indent=2)

    # 11) Plot and save training/validation loss curves.
    plot_history(history, os.path.join(OUT_DIR, "training_loss.png"))

    # 12) Evaluate on TEST set (convert predictions back to original units).
    y_test_pred_sc = model.predict(X_test_sc, verbose=0)   # Predict in scaled space.
    y_test_pred    = y_scaler.inverse_transform(y_test_pred_sc)  # Invert scaling.

    # Compute metrics (MAE, RMSE, R²) on test set.
    metrics = metrics_report(y_test, y_test_pred, TARGETS)

    # 13) Check if model meets adequacy thresholds.
    thresholds = {
        "min_r2_overall": 0.97,
        "max_mae_overall": 0.035,
        "per_target_min_r2": {"Cl":0.98, "Cd":0.70, "Cdp":0.70, "Cm":0.60}
    }
    verdict = assess_accuracy(metrics, thresholds)

    print("=== Adequacy verdict ===")
    if verdict["adequate"]:
        print("✅ Model accuracy is ADEQUATE for the configured thresholds.")
    else:
        print("❌ Model accuracy is NOT adequate for the configured thresholds.")
        for line in verdict["fail_reasons"]:
            print(" ", line)

    # Save metrics and adequacy verdict to disk.
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "verdict": verdict}, f, indent=2)

    # 14) Create scatter plots of predicted vs true coefficients for the test set.
    plot_pred_vs_true(y_test, y_test_pred, TARGETS, OUT_DIR)

    # 15) Print one example test case with clean floating-point values.
    idx = 0  # Use the first test sample as an example.
    example_inputs = {k: float(v) for k, v in zip(NUM_FEATURES, X_num_test[idx].tolist())}
    true_vals      = {k: float(v) for k, v in zip(TARGETS,     y_test[idx].tolist())}
    pred_vals      = {k: float(v) for k, v in zip(TARGETS,     y_test_pred[idx].tolist())}

    print("Example Airfoil:", airfoil_test[idx, 0])
    print("Example numeric inputs (original units):", to_py_floats(example_inputs, 6))
    print("True  (original units):", to_py_floats(true_vals, 6))
    print("Pred  (original units):", to_py_floats(pred_vals, 6))
    print("\nArtifacts saved in:", os.path.abspath(OUT_DIR))


# This block runs only if the file is executed as a script (not imported as a module).
if __name__ == "__main__":
    # Optional: you could uncomment this to silence some TensorFlow oneDNN warnings.
    # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    main()  # Call the main() function to start the whole process.
