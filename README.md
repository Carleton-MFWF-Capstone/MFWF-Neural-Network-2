# Aerodynamic Coefficient Predictor

MLP-based machine learning tool for predicting:

- Cl (Lift Coefficient)
- Cd (Drag Coefficient)
- Cdp (Profile Drag Coefficient)
- Cm (Pitching Moment Coefficient)

## Features

- Python and TensorFlow MLP model
- One-hot encoded airfoil geometry
- Streamlit interactive web interface
- Batch prediction script for Excel inputs

## Data Files

The training and testing workbooks are intentionally not tracked in Git.
Place these files in the project folder when running locally:

- `Complete Training Dataset (All NACA Airfoils).xlsx`
- `Testing Dataset - NACA 4 Digit Airfoils.xlsx`

## Train Model

```bash
python train_mlp_aero.py
```

## Run Web Interface

```bash
python -m streamlit run streamlit_app.py
```
