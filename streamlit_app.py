# ============================================================
# streamlit_app.py
# ------------------------------------------------------------
# Aerodynamic Coefficient Tool (MLP)
# - 2D & 3D Plotly visualizations
# - Smooth animations using Plotly frames (no Python sleep loops)
# - FIXED 3D Surface animation (redraw=True + correct frame updates)
# - User-controlled animation range for 3D "Animate Variable"
# - Axis lock options during animation:
#     Unlocked | Lock to current selected ranges | Lock to maximum needed across frames
# - Min/Max table under 2D plot (and Z min/max under 3D)
# - Compact layout with reduced white space
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow import keras
import plotly.graph_objects as go

# -----------------------------
# Constants
# -----------------------------
TARGETS = ["Cl", "Cd", "Cdp", "Cm"]
VAR_OPTIONS = ["Alpha", "Re", "Top_Xtr", "Bot_Xtr"]
VAR_TO_IDX = {"Re": 0, "Alpha": 1, "Top_Xtr": 2, "Bot_Xtr": 3}

COLOR_MAP = {"Cl": "#1f77b4", "Cd": "#ff7f0e", "Cdp": "#2ca02c", "Cm": "#d62728"}

DATA_FILE_DEFAULT = "Complete Training Dataset (All NACA Airfoils).xlsx"
MODELS_DIR = "models"


# --------------------------------------------------
# Load model, preprocessors, and dataset
# --------------------------------------------------
@st.cache_resource
def load_model_and_preprocessors():
    model = keras.models.load_model(os.path.join(MODELS_DIR, "aero_mlp.keras"), compile=False)
    x_scaler = joblib.load(os.path.join(MODELS_DIR, "x_scaler.pkl"))
    y_scaler = joblib.load(os.path.join(MODELS_DIR, "y_scaler.pkl"))
    airfoil_encoder = joblib.load(os.path.join(MODELS_DIR, "airfoil_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    return model, x_scaler, y_scaler, airfoil_encoder, meta


def categorize_airfoil(name: str) -> str:
    s = str(name).strip()
    sl = s.lower()
    if sl.endswith("-jf"):
        return "Joukowski"
    if sl.startswith("naca"):
        work = sl[4:]
    else:
        work = sl
    digits = "".join(ch for ch in work if ch.isdigit())
    if not digits:
        return "Other"
    if digits.startswith("16"):
        return "NACA 16 Series"
    if digits[0] == "6":
        return "NACA 6 Series"
    if digits[0] == "7":
        return "NACA 7 Series"
    if len(digits) == 4:
        return "NACA 4 digit"
    if len(digits) == 5:
        return "NACA 5 digit"
    return "Other"


@st.cache_data
def load_training_data(data_file, num_features, target_names):
    df = pd.read_excel(data_file)
    df["Airfoil"] = df["Airfoil"].astype(str)

    cols = num_features + target_names
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols + ["Airfoil"])

    ranges = {col: (float(df[col].min()), float(df[col].max())) for col in num_features}
    all_airfoils = sorted(df["Airfoil"].unique().tolist())
    return df, ranges, all_airfoils


def get_airfoils_for_category(all_airfoils, category: str):
    if category == "All":
        return list(all_airfoils)
    return [a for a in all_airfoils if categorize_airfoil(a) == category]


# --------------------------------------------------
# Inference helpers
# --------------------------------------------------
def predict_coeffs(model, x_scaler, y_scaler, airfoil_encoder,
                   airfoil_name, Re, Alpha, Top_Xtr, Bot_Xtr):
    x_num = np.array([[Re, Alpha, Top_Xtr, Bot_Xtr]], dtype=np.float32)
    airfoil_arr = np.array([[airfoil_name]])

    x_num_sc = x_scaler.transform(x_num)
    x_cat = airfoil_encoder.transform(airfoil_arr)
    x_input = np.hstack([x_num_sc, x_cat]).astype(np.float32)

    y_pred_sc = model.predict(x_input, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_sc)[0]
    return {name: float(val) for name, val in zip(TARGETS, y_pred)}


def sweep_alpha_curve(model, x_scaler, y_scaler, airfoil_encoder,
                      airfoil_name, Re, Top_Xtr, Bot_Xtr,
                      alpha_min, alpha_max, num_points=240):
    alphas = np.linspace(alpha_min, alpha_max, num_points, dtype=np.float32)

    X_num = np.column_stack([
        np.full_like(alphas, Re, dtype=np.float32),
        alphas,
        np.full_like(alphas, Top_Xtr, dtype=np.float32),
        np.full_like(alphas, Bot_Xtr, dtype=np.float32),
    ])

    airfoil_arr = np.array([[airfoil_name]] * num_points)
    X_num_sc = x_scaler.transform(X_num)
    X_cat = airfoil_encoder.transform(airfoil_arr)
    X_input = np.hstack([X_num_sc, X_cat]).astype(np.float32)

    Y_sc = model.predict(X_input, verbose=0)
    Y = y_scaler.inverse_transform(Y_sc)

    return pd.DataFrame({
        "Alpha": alphas,
        "Cl":  Y[:, 0],
        "Cd":  Y[:, 1],
        "Cdp": Y[:, 2],
        "Cm":  Y[:, 3],
    })


def compute_minmax_table_2d(df_curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        i_min = int(df_curve[t].idxmin())
        i_max = int(df_curve[t].idxmax())
        rows.append({
            "Coefficient": t,
            "Min": float(df_curve.loc[i_min, t]),
            "Alpha at Min (deg)": float(df_curve.loc[i_min, "Alpha"]),
            "Max": float(df_curve.loc[i_max, t]),
            "Alpha at Max (deg)": float(df_curve.loc[i_max, "Alpha"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------
# Compact slider + typed input control
# --------------------------------------------------
def slider_with_value(label, min_val, max_val, value, step, key, fmt="%.4f", help_text=None):
    col_s, col_n = st.columns([3.0, 1.2], vertical_alignment="center")
    slider_key = f"{key}_sl"
    num_key = f"{key}_num"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = float(value)
    if num_key not in st.session_state:
        st.session_state[num_key] = float(value)

    def sync_from_slider():
        st.session_state[num_key] = float(st.session_state[slider_key])

    def sync_from_num():
        st.session_state[slider_key] = float(st.session_state[num_key])

    with col_s:
        st.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(st.session_state[slider_key]),
            step=float(step),
            key=slider_key,
            on_change=sync_from_slider,
            help=help_text,
        )
    with col_n:
        st.number_input(
            " ",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(st.session_state[num_key]),
            step=float(step),
            format=fmt,
            key=num_key,
            on_change=sync_from_num,
            label_visibility="collapsed",
        )
    return float(st.session_state[slider_key])


# --------------------------------------------------
# Plotly animation controls (below plot, no overlap)
# --------------------------------------------------
def add_animation_controls_below(fig, frame_duration_ms=40, slider_title="Frame", redraw=False):
    """
    Places Play/Pause + slider below the plot using paper coords (negative y).
    For 3D Surface traces, redraw=True is typically REQUIRED to see changes.
    """
    if not fig.frames:
        return fig

    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            x=0.02, y=-0.08,
            xanchor="left", yanchor="top",
            direction="right",
            pad=dict(r=8, t=0),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_duration_ms, redraw=redraw),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=redraw), mode="immediate")],
                ),
            ],
        )
    ]

    sliders = [
        dict(
            x=0.02, y=-0.16,
            xanchor="left", yanchor="top",
            len=0.96,
            pad=dict(t=0, b=0),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            currentvalue=dict(prefix=f"{slider_title}: "),
            steps=[
                dict(
                    method="animate",
                    args=[
                        [fr.name],
                        dict(
                            frame=dict(duration=frame_duration_ms, redraw=redraw),
                            transition=dict(duration=0),
                            mode="immediate",
                        ),
                    ],
                    label=str(i + 1),
                )
                for i, fr in enumerate(fig.frames)
            ],
        )
    ]

    fig.update_layout(updatemenus=updatemenus, sliders=sliders)
    return fig


# --------------------------------------------------
# 2D Figures + Animations
# --------------------------------------------------
def build_2d_static(df_curve, title, axis_lock_mode="Unlocked"):
    fig = go.Figure()
    for t in TARGETS:
        fig.add_trace(go.Scatter(
            x=df_curve["Alpha"], y=df_curve[t],
            mode="lines", name=t,
            line=dict(color=COLOR_MAP[t]),
        ))

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Angle of Attack α (deg)",
        yaxis_title="Coefficient value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=45, b=95),
        uirevision="keep-2d",
    )

    if axis_lock_mode != "Unlocked":
        x_rng = (float(df_curve["Alpha"].min()), float(df_curve["Alpha"].max()))
        y_all = np.concatenate([df_curve[t].values for t in TARGETS], axis=0)
        y_rng = (float(np.nanmin(y_all)), float(np.nanmax(y_all)))
        fig.update_xaxes(range=list(x_rng))
        fig.update_yaxes(range=list(y_rng))

    return fig


def build_2d_alpha_highlight_animation(df_curve, title_base, n_frames=60, fps=30, axis_lock_mode="Unlocked"):
    alphas = df_curve["Alpha"].values
    a_min, a_max = float(alphas.min()), float(alphas.max())
    alpha_anim = np.linspace(a_min, a_max, int(n_frames))

    fig = go.Figure()

    # 4 coefficient lines
    for t in TARGETS:
        fig.add_trace(go.Scatter(
            x=alphas, y=df_curve[t].values,
            mode="lines", name=t,
            line=dict(color=COLOR_MAP[t]),
        ))

    # y range for vertical line
    y_all = np.concatenate([df_curve[t].values for t in TARGETS], axis=0)
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))

    # vertical line trace
    fig.add_trace(go.Scatter(
        x=[alpha_anim[0], alpha_anim[0]],
        y=[y_min, y_max],
        mode="lines",
        line=dict(color="black", width=1, dash="dash"),
        showlegend=False,
    ))

    # markers at highlight alpha
    for t in TARGETS:
        fig.add_trace(go.Scatter(
            x=[alpha_anim[0]],
            y=[float(np.interp(alpha_anim[0], alphas, df_curve[t].values))],
            mode="markers",
            marker=dict(size=7, color=COLOR_MAP[t]),
            showlegend=False,
        ))

    frames = []
    for k, a in enumerate(alpha_anim):
        ys = [float(np.interp(a, alphas, df_curve[t].values)) for t in TARGETS]
        frame_data = [go.Scatter(x=[a, a], y=[y_min, y_max])]
        for yi in ys:
            frame_data.append(go.Scatter(x=[a], y=[yi]))

        frames.append(go.Frame(
            name=f"f{k}",
            data=frame_data,
            traces=[4, 5, 6, 7, 8],
            layout=dict(title=dict(text=f"{title_base} (α={a:.2f}°)", x=0.5, xanchor="center")),
        ))

    fig.frames = frames
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title_base, x=0.5, xanchor="center"),
        xaxis_title="Angle of Attack α (deg)",
        yaxis_title="Coefficient value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=45, b=140),
        uirevision="keep-2d",
    )

    if axis_lock_mode != "Unlocked":
        fig.update_xaxes(range=[a_min, a_max])
        fig.update_yaxes(range=[y_min, y_max])

    frame_ms = max(15, int(1000 / max(1, int(fps))))
    fig = add_animation_controls_below(fig, frame_duration_ms=frame_ms, slider_title="α frame", redraw=False)
    return fig


def build_2d_variable_animation(model, x_scaler, y_scaler, airfoil_encoder,
                                airfoil_name, anim_var, anim_vals,
                                Re, Top_Xtr, Bot_Xtr,
                                alpha_min, alpha_max,
                                num_points=240,
                                fps=30,
                                axis_lock_mode="Unlocked"):
    alphas = np.linspace(alpha_min, alpha_max, num_points, dtype=np.float32)

    curves = []
    y_global_min, y_global_max = np.inf, -np.inf

    for v in anim_vals:
        new_Re, new_Top, new_Bot = float(Re), float(Top_Xtr), float(Bot_Xtr)
        if anim_var == "Re":
            new_Re = float(v)
        elif anim_var == "Top_Xtr":
            new_Top = float(v)
        elif anim_var == "Bot_Xtr":
            new_Bot = float(v)

        X_num = np.column_stack([
            np.full_like(alphas, new_Re, dtype=np.float32),
            alphas,
            np.full_like(alphas, new_Top, dtype=np.float32),
            np.full_like(alphas, new_Bot, dtype=np.float32),
        ])

        airfoil_arr = np.array([[airfoil_name]] * len(alphas))
        X_num_sc = x_scaler.transform(X_num)
        X_cat = airfoil_encoder.transform(airfoil_arr)
        X_input = np.hstack([X_num_sc, X_cat]).astype(np.float32)

        Y_sc = model.predict(X_input, verbose=0)
        Y = y_scaler.inverse_transform(Y_sc)

        yk = {t: Y[:, i] for i, t in enumerate(TARGETS)}
        curves.append(yk)

        if axis_lock_mode == "Lock to maximum needed across frames":
            allv = np.concatenate([yk[t] for t in TARGETS], axis=0)
            y_global_min = min(y_global_min, float(np.nanmin(allv)))
            y_global_max = max(y_global_max, float(np.nanmax(allv)))

    fig = go.Figure()
    y0 = curves[0]
    for t in TARGETS:
        fig.add_trace(go.Scatter(
            x=alphas, y=y0[t],
            mode="lines", name=t,
            line=dict(color=COLOR_MAP[t]),
        ))

    frames = []
    for k, v in enumerate(anim_vals):
        yk = curves[k]
        frames.append(go.Frame(
            name=f"v{k}",
            data=[go.Scatter(y=yk[t]) for t in TARGETS],
            traces=[0, 1, 2, 3],
            layout=dict(title=dict(
                text=f"Coefficients vs α — {airfoil_name} ({anim_var}={float(v):.6g})",
                x=0.5, xanchor="center"
            )),
        ))

    fig.frames = frames
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"Coefficients vs α — {airfoil_name} ({anim_var}={float(anim_vals[0]):.6g})",
                   x=0.5, xanchor="center"),
        xaxis_title="Angle of Attack α (deg)",
        yaxis_title="Coefficient value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=45, b=140),
        uirevision="keep-2d",
    )

    x_rng = (float(alpha_min), float(alpha_max))
    if axis_lock_mode == "Lock to current selected ranges":
        all0 = np.concatenate([y0[t] for t in TARGETS], axis=0)
        y_rng = (float(np.nanmin(all0)), float(np.nanmax(all0)))
        fig.update_xaxes(range=list(x_rng))
        fig.update_yaxes(range=list(y_rng))
    elif axis_lock_mode == "Lock to maximum needed across frames":
        fig.update_xaxes(range=list(x_rng))
        fig.update_yaxes(range=[float(y_global_min), float(y_global_max)])

    frame_ms = max(15, int(1000 / max(1, int(fps))))
    fig = add_animation_controls_below(fig, frame_duration_ms=frame_ms, slider_title=anim_var, redraw=False)
    return fig


# --------------------------------------------------
# 3D helpers
# --------------------------------------------------
def make_feature_grid(var_x, var_y, x_vals, y_vals, constants):
    Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="xy")
    n = Xg.size

    base = np.zeros((n, 4), dtype=np.float32)
    base[:, 0] = float(constants["Re"])
    base[:, 1] = float(constants["Alpha"])
    base[:, 2] = float(constants["Top_Xtr"])
    base[:, 3] = float(constants["Bot_Xtr"])

    base[:, VAR_TO_IDX[var_x]] = Xg.reshape(-1).astype(np.float32)
    base[:, VAR_TO_IDX[var_y]] = Yg.reshape(-1).astype(np.float32)
    return Xg, Yg, base


def predict_grid(model, x_scaler, y_scaler, airfoil_encoder,
                 airfoil_name, var_x, var_y, x_vals, y_vals, constants):
    Xg, Yg, X_num = make_feature_grid(var_x, var_y, x_vals, y_vals, constants)
    airfoil_arr = np.array([[airfoil_name]] * X_num.shape[0])

    X_num_sc = x_scaler.transform(X_num)
    X_cat = airfoil_encoder.transform(airfoil_arr)
    X_input = np.hstack([X_num_sc, X_cat]).astype(np.float32)

    Y_sc = model.predict(X_input, verbose=0)
    Y = y_scaler.inverse_transform(Y_sc)
    preds = {t: Y[:, i].reshape(Yg.shape) for i, t in enumerate(TARGETS)}
    return Xg, Yg, preds


def build_3d_static_surface(model, x_scaler, y_scaler, airfoil_encoder,
                            airfoil_name, var_x, var_y, x_vals, y_vals, constants,
                            z_coeff, color_coeff, axis_lock_mode="Unlocked"):
    Xg, Yg, preds = predict_grid(
        model, x_scaler, y_scaler, airfoil_encoder,
        airfoil_name, var_x, var_y, x_vals, y_vals, constants
    )
    Zg = preds[z_coeff]
    Cg = Zg if color_coeff == "(same as Z)" else preds[color_coeff]
    color_title = z_coeff if color_coeff == "(same as Z)" else color_coeff

    fig = go.Figure(data=[
        go.Surface(
            x=Xg, y=Yg, z=Zg,
            surfacecolor=Cg,
            colorscale="Viridis",
            colorbar=dict(title=color_title),
        )
    ])

    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"3D Surface: {z_coeff} over ({var_x}, {var_y}) — {airfoil_name}",
                   x=0.5, xanchor="center"),
        scene=dict(xaxis_title=var_x, yaxis_title=var_y, zaxis_title=z_coeff),
        margin=dict(l=0, r=0, t=45, b=95),
        uirevision="keep-3d",
    )

    if axis_lock_mode != "Unlocked":
        zmin, zmax = float(np.nanmin(Zg)), float(np.nanmax(Zg))
        fig.update_layout(scene=dict(
            xaxis=dict(range=[float(np.min(x_vals)), float(np.max(x_vals))]),
            yaxis=dict(range=[float(np.min(y_vals)), float(np.max(y_vals))]),
            zaxis=dict(range=[zmin, zmax]),
        ))

    return fig, Zg


def build_3d_variable_animation(model, x_scaler, y_scaler, airfoil_encoder,
                                airfoil_name, var_x, var_y, x_vals, y_vals,
                                constants_base, anim_var, anim_vals,
                                z_coeff, color_coeff,
                                fps=25,
                                axis_lock_mode="Unlocked"):
    """
    FIXED 3D Surface animation:
      - frames update trace 0 explicitly (Surface)
      - update only z and surfacecolor per frame
      - redraw=True in animation controls so WebGL surface repaints
      - uirevision preserves user's camera angle during animation
    """
    # Initial frame prediction
    c0 = dict(constants_base)
    c0[anim_var] = float(anim_vals[0])

    Xg0, Yg0, preds0 = predict_grid(
        model, x_scaler, y_scaler, airfoil_encoder,
        airfoil_name, var_x, var_y, x_vals, y_vals, c0
    )
    Z0 = preds0[z_coeff]
    C0 = Z0 if color_coeff == "(same as Z)" else preds0[color_coeff]
    color_title = z_coeff if color_coeff == "(same as Z)" else color_coeff

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=Xg0, y=Yg0, z=Z0,
            surfacecolor=C0,
            colorscale="Viridis",
            colorbar=dict(title=color_title),
        )
    )

    # Axis lock computation
    x_rng = [float(np.min(x_vals)), float(np.max(x_vals))]
    y_rng = [float(np.min(y_vals)), float(np.max(y_vals))]

    z0min, z0max = float(np.nanmin(Z0)), float(np.nanmax(Z0))
    z_global_min, z_global_max = z0min, z0max

    if axis_lock_mode == "Lock to maximum needed across frames":
        for v in anim_vals[1:]:
            ck = dict(constants_base)
            ck[anim_var] = float(v)
            _, _, preds = predict_grid(
                model, x_scaler, y_scaler, airfoil_encoder,
                airfoil_name, var_x, var_y, x_vals, y_vals, ck
            )
            Zk = preds[z_coeff]
            z_global_min = min(z_global_min, float(np.nanmin(Zk)))
            z_global_max = max(z_global_max, float(np.nanmax(Zk)))

    frames = []
    for k, v in enumerate(anim_vals):
        ck = dict(constants_base)
        ck[anim_var] = float(v)

        Xg, Yg, preds = predict_grid(
            model, x_scaler, y_scaler, airfoil_encoder,
            airfoil_name, var_x, var_y, x_vals, y_vals, ck
        )
        Zk = preds[z_coeff]
        Ck = Zk if color_coeff == "(same as Z)" else preds[color_coeff]

        frames.append(
            go.Frame(
                name=f"f{k}",
                data=[go.Surface(z=Zk, surfacecolor=Ck)],
                traces=[0],
                layout=dict(
                    title=dict(
                        text=f"3D Surface: {z_coeff} — {airfoil_name} ({anim_var}={float(v):.6g})",
                        x=0.5, xanchor="center"
                    )
                )
            )
        )

    fig.frames = frames

    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"3D Surface: {z_coeff} — {airfoil_name} ({anim_var}={float(anim_vals[0]):.6g})",
                   x=0.5, xanchor="center"),
        scene=dict(xaxis_title=var_x, yaxis_title=var_y, zaxis_title=z_coeff),
        margin=dict(l=0, r=0, t=45, b=140),
        uirevision="keep-3d",
    )

    # Apply axis locks (do not set camera)
    if axis_lock_mode == "Lock to current selected ranges":
        fig.update_layout(scene=dict(
            xaxis=dict(range=x_rng),
            yaxis=dict(range=y_rng),
            zaxis=dict(range=[z0min, z0max]),
        ))
    elif axis_lock_mode == "Lock to maximum needed across frames":
        fig.update_layout(scene=dict(
            xaxis=dict(range=x_rng),
            yaxis=dict(range=y_rng),
            zaxis=dict(range=[float(z_global_min), float(z_global_max)]),
        ))

    frame_ms = max(25, int(1000 / max(1, int(fps))))
    fig = add_animation_controls_below(
        fig,
        frame_duration_ms=frame_ms,
        slider_title=anim_var,
        redraw=True,  # critical for WebGL Surface updates
    )
    return fig


# --------------------------------------------------
# Main UI
# --------------------------------------------------
def main():
    st.set_page_config(page_title="Aerodynamic Coefficient Tool", layout="wide")

    # Compact CSS
    st.markdown(
        """
        <style>
          section.main > div.block-container {
            padding-top: 0.05rem;
            padding-bottom: 0.05rem;
            max-width: 1500px;
          }
          h1 { margin: 0.0rem 0 0.10rem 0 !important; }
          h2, h3 { margin: 0.10rem 0 0.10rem 0 !important; }
          hr { margin: 0.25rem 0 !important; }
          div[data-testid="stVerticalBlock"] { gap: 0.35rem; }
          div[data-testid="stHorizontalBlock"] { gap: 0.6rem; }
          button[role="tab"] { padding-top: 0.25rem !important; padding-bottom: 0.25rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Aerodynamic Coefficient Tool")

    model, x_scaler, y_scaler, airfoil_encoder, meta = load_model_and_preprocessors()
    num_feature_names = meta.get("num_features", ["Re", "Alpha", "Top_Xtr", "Bot_Xtr"])
    target_names = meta.get("targets", TARGETS)

    df_data, ranges, all_airfoils = load_training_data(DATA_FILE_DEFAULT, num_feature_names, target_names)

    tab_pred, tab_inverse, tab_info = st.tabs([
        "Aerodynamic Coefficient Predictor",
        "Inverse Design: Suggest Airfoils",
        "Variable & Model Info"
    ])

    with tab_pred:
        st.subheader("Aerodynamic Coefficient Predictor")
        st.caption("2D/3D interactive visualization with smooth Plotly frame animations (including fixed 3D Surface animation).")

        # 3-column layout
        col_inputs, col_controls, col_plot = st.columns([1.05, 1.1, 1.85], vertical_alignment="top")

        # ---------- Inputs ----------
        with col_inputs:
            st.markdown("### Airfoil")

            canonical_cats = [
                "NACA 4 digit", "NACA 5 digit", "NACA 6 Series",
                "NACA 7 Series", "NACA 16 Series", "Joukowski", "Other",
            ]
            total_n = len(all_airfoils)
            labels = [f"All ({total_n})"]
            label_to_cat = {f"All ({total_n})": "All"}
            for cat in canonical_cats:
                ncat = sum(categorize_airfoil(a) == cat for a in all_airfoils)
                if ncat > 0:
                    lab = f"{cat} ({ncat})"
                    labels.append(lab)
                    label_to_cat[lab] = cat

            selected_label = st.selectbox("Family/category", labels, key="cat_sel")
            selected_category = label_to_cat[selected_label]
            airfoil_options = sorted(get_airfoils_for_category(all_airfoils, selected_category))
            airfoil_name = st.selectbox("Airfoil ID", airfoil_options, key="airfoil_sel")

            st.divider()
            st.markdown("### Inputs")

            Re = slider_with_value(
                "Re", *ranges["Re"],
                value=(ranges["Re"][0] + ranges["Re"][1]) / 2,
                step=max((ranges["Re"][1] - ranges["Re"][0]) / 120, 1.0),
                key="Re", fmt="%.1f"
            )
            Alpha = slider_with_value(
                "Alpha (deg)", *ranges["Alpha"],
                value=(ranges["Alpha"][0] + ranges["Alpha"][1]) / 2,
                step=0.25, key="Alpha", fmt="%.2f"
            )
            Top_Xtr = slider_with_value(
                "Top_Xtr (x/c)", *ranges["Top_Xtr"],
                value=(ranges["Top_Xtr"][0] + ranges["Top_Xtr"][1]) / 2,
                step=0.01, key="TopXtr", fmt="%.4f"
            )
            Bot_Xtr = slider_with_value(
                "Bot_Xtr (x/c)", *ranges["Bot_Xtr"],
                value=(ranges["Bot_Xtr"][0] + ranges["Bot_Xtr"][1]) / 2,
                step=0.01, key="BotXtr", fmt="%.4f"
            )

            st.divider()
            plot_mode = st.radio("Graph type", ["2D", "3D"], horizontal=True, key="plot_mode")

        # ---------- Controls ----------
        with col_controls:
            st.markdown("### Controls")

            axis_lock_mode = st.selectbox(
                "Axis lock during animation",
                ["Unlocked", "Lock to current selected ranges", "Lock to maximum needed across frames"],
                index=1,
                key="axis_lock_mode",
            )

            if plot_mode == "2D":
                st.divider()
                st.markdown("#### 2D Sweep (α range)")
                a_min = st.number_input("α min", value=float(ranges["Alpha"][0]), step=0.5, format="%.2f", key="a_min")
                a_max = st.number_input("α max", value=float(ranges["Alpha"][1]), step=0.5, format="%.2f", key="a_max")

                st.divider()
                st.markdown("#### 2D Animation")
                anim2d_mode = st.radio(
                    "Mode",
                    ["Static", "Animate α (highlight)", "Animate variable (curves)"],
                    index=0,
                    key="anim2d_mode",
                )

                fps_2d = st.slider("FPS", 10, 60, 30, 5, key="fps_2d")
                frames_2d = st.slider("Frames", 10, 120, 60, 10, key="frames_2d") if anim2d_mode != "Static" else 0

                anim_var_2d = None
                if anim2d_mode == "Animate variable (curves)":
                    anim_var_2d = st.selectbox("Animate variable", ["Re", "Top_Xtr", "Bot_Xtr"], key="anim_var_2d")

            else:
                st.divider()
                st.markdown("#### 3D Axes")
                var_x = st.selectbox("X-axis", VAR_OPTIONS, index=0, key="var_x")
                var_y = st.selectbox("Y-axis", VAR_OPTIONS, index=2, key="var_y")
                z_coeff = st.selectbox("Z coefficient", TARGETS, index=0, key="z_coeff")
                color_coeff = st.selectbox("Color map", ["(same as Z)"] + TARGETS, index=0, key="color_coeff")

                st.divider()
                st.markdown("#### 3D Ranges + Resolution")
                x0min, x0max = ranges[var_x]
                y0min, y0max = ranges[var_y]

                x_min = st.number_input(f"{var_x} min", value=float(x0min), format="%.6f", key="x_min")
                x_max = st.number_input(f"{var_x} max", value=float(x0max), format="%.6f", key="x_max")
                y_min = st.number_input(f"{var_y} min", value=float(y0min), format="%.6f", key="y_min")
                y_max = st.number_input(f"{var_y} max", value=float(y0max), format="%.6f", key="y_max")
                res = st.slider("Resolution", 20, 120, 60, 10, key="res")

                st.divider()
                st.markdown("#### 3D Animation")
                anim3d_mode = st.radio(
                    "Mode",
                    ["Static", "Animate variable (surface)"],
                    index=0,
                    key="anim3d_mode",
                )
                fps_3d = st.slider("FPS", 10, 60, 25, 5, key="fps_3d")
                frames_3d = st.slider("Frames", 10, 80, 25, 5, key="frames_3d") if anim3d_mode != "Static" else 0

                held = [v for v in VAR_OPTIONS if v not in [var_x, var_y]]
                anim_var_3d = None
                anim_min = None
                anim_max = None
                if anim3d_mode != "Static":
                    if not held:
                        st.warning("No available variable to animate (choose different X/Y).")
                    else:
                        anim_var_3d = st.selectbox("Animate variable", held, key="anim_var_3d")

                        # User-controlled animation range (fixes "range too small to see change")
                        vmin_ds, vmax_ds = ranges[anim_var_3d]
                        st.markdown("##### Animation range")
                        anim_min = st.number_input(
                            f"{anim_var_3d} anim min",
                            min_value=float(vmin_ds),
                            max_value=float(vmax_ds),
                            value=float(vmin_ds),
                            format="%.6f",
                            key="anim3d_min",
                        )
                        anim_max = st.number_input(
                            f"{anim_var_3d} anim max",
                            min_value=float(vmin_ds),
                            max_value=float(vmax_ds),
                            value=float(vmax_ds),
                            format="%.6f",
                            key="anim3d_max",
                        )

        # ---------- Plot + outputs ----------
        with col_plot:
            st.markdown("### Output")

            pred = predict_coeffs(model, x_scaler, y_scaler, airfoil_encoder,
                                  airfoil_name, Re, Alpha, Top_Xtr, Bot_Xtr)

            mcols = st.columns(4)
            for i, t in enumerate(target_names):
                mcols[i].metric(t, f"{pred[t]:.5f}")

            st.divider()

            config = {"displaylogo": False, "responsive": True}

            if plot_mode == "2D":
                if a_max < a_min:
                    st.error("α max must be >= α min.")
                    st.stop()

                df_curve = sweep_alpha_curve(
                    model, x_scaler, y_scaler, airfoil_encoder,
                    airfoil_name, Re, Top_Xtr, Bot_Xtr,
                    float(a_min), float(a_max),
                    num_points=240,
                )

                title_base = f"Coefficients vs α — {airfoil_name} (Re={Re:.0f})"

                if anim2d_mode == "Static":
                    fig = build_2d_static(df_curve, title_base, axis_lock_mode=axis_lock_mode)
                elif anim2d_mode == "Animate α (highlight)":
                    fig = build_2d_alpha_highlight_animation(
                        df_curve, title_base,
                        n_frames=int(frames_2d),
                        fps=int(fps_2d),
                        axis_lock_mode=axis_lock_mode,
                    )
                else:
                    vmin, vmax = ranges[anim_var_2d]
                    anim_vals = np.linspace(vmin, vmax, int(frames_2d)).astype(np.float32)
                    fig = build_2d_variable_animation(
                        model, x_scaler, y_scaler, airfoil_encoder,
                        airfoil_name=airfoil_name,
                        anim_var=anim_var_2d,
                        anim_vals=anim_vals,
                        Re=Re, Top_Xtr=Top_Xtr, Bot_Xtr=Bot_Xtr,
                        alpha_min=float(a_min),
                        alpha_max=float(a_max),
                        num_points=240,
                        fps=int(fps_2d),
                        axis_lock_mode=axis_lock_mode,
                    )

                st.plotly_chart(fig, use_container_width=True, config=config)

                # Min/max table under plot
                st.markdown("#### Min / Max over α sweep")
                df_mm = compute_minmax_table_2d(df_curve)
                df_show = df_mm.copy()
                df_show["Min"] = df_show["Min"].map(lambda v: f"{v:.5f}")
                df_show["Max"] = df_show["Max"].map(lambda v: f"{v:.5f}")
                df_show["Alpha at Min (deg)"] = df_show["Alpha at Min (deg)"].map(lambda v: f"{v:.2f}")
                df_show["Alpha at Max (deg)"] = df_show["Alpha at Max (deg)"].map(lambda v: f"{v:.2f}")
                st.table(df_show)

            else:
                # 3D
                if var_x == var_y:
                    st.error("X-axis and Y-axis must be different.")
                    st.stop()
                if x_max <= x_min or y_max <= y_min:
                    st.error("Max must be greater than min for both axes.")
                    st.stop()

                constants = {"Re": Re, "Alpha": Alpha, "Top_Xtr": Top_Xtr, "Bot_Xtr": Bot_Xtr}
                x_vals = np.linspace(float(x_min), float(x_max), int(res), dtype=np.float32)
                y_vals = np.linspace(float(y_min), float(y_max), int(res), dtype=np.float32)

                if anim3d_mode == "Static" or anim_var_3d is None:
                    fig3d, Zg = build_3d_static_surface(
                        model, x_scaler, y_scaler, airfoil_encoder,
                        airfoil_name, var_x, var_y, x_vals, y_vals, constants,
                        z_coeff=z_coeff, color_coeff=color_coeff,
                        axis_lock_mode=axis_lock_mode,
                    )
                    st.plotly_chart(fig3d, use_container_width=True, config=config)

                    # Z min/max under plot
                    zmin, zmax = float(np.nanmin(Zg)), float(np.nanmax(Zg))
                    st.markdown("#### Z min / max on surface")
                    st.write(f"**{z_coeff} min:** {zmin:.5f}   |   **{z_coeff} max:** {zmax:.5f}")

                else:
                    if anim_max is None or anim_min is None:
                        st.stop()
                    if anim_max <= anim_min:
                        st.error("Animation max must be greater than animation min.")
                        st.stop()

                    anim_vals = np.linspace(float(anim_min), float(anim_max), int(frames_3d)).astype(np.float32)

                    fig_anim = build_3d_variable_animation(
                        model, x_scaler, y_scaler, airfoil_encoder,
                        airfoil_name=airfoil_name,
                        var_x=var_x, var_y=var_y,
                        x_vals=x_vals, y_vals=y_vals,
                        constants_base=constants,
                        anim_var=anim_var_3d,
                        anim_vals=anim_vals,
                        z_coeff=z_coeff,
                        color_coeff=color_coeff,
                        fps=int(fps_3d),
                        axis_lock_mode=axis_lock_mode,
                    )
                    st.plotly_chart(fig_anim, use_container_width=True, config=config)

                    # Show Z min/max for the INITIAL frame
                    c0 = dict(constants)
                    c0[anim_var_3d] = float(anim_vals[0])
                    _, _, preds0 = predict_grid(
                        model, x_scaler, y_scaler, airfoil_encoder,
                        airfoil_name, var_x, var_y, x_vals, y_vals, c0
                    )
                    Z0 = preds0[z_coeff]
                    z0min, z0max = float(np.nanmin(Z0)), float(np.nanmax(Z0))
                    st.markdown("#### Z min / max (initial frame)")
                    st.write(f"**{z_coeff} min:** {z0min:.5f}   |   **{z_coeff} max:** {z0max:.5f}")

    with tab_inverse:
        st.subheader("Inverse Design: Suggest Airfoils for Desired Outputs")
        st.write("No updates applied in this tab in this iteration.")

    with tab_info:
        st.subheader("Variable & Model Info")
        st.markdown(
            """
            **Key animation note (3D):** Plotly WebGL `Surface` animations require `redraw=True`
            so the surface repaints each frame. This app uses Plotly frames for smooth animation
            without Python-side sleep loops.
            """
        )


if __name__ == "__main__":
    main()
