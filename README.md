# Aerodynamic Coefficient Predictor

MLP-based machine learning tool for predicting:
- Cl (Lift Coefficient)
- Cd (Drag Coefficient)
- Cdp (Profile Drag Coefficient)
- Cm (Pitching Moment Coefficient)

## Features
- Python + TensorFlow MLP model
- One-hot encoded airfoil geometry
- Streamlit interactive web interface
- Inverse design capability

## NACA Airfoil Dataset
[Complete Training Dataset (All NACA Airfoils).xlsx](https://github.com/user-attachments/files/24954759/Complete.Training.Dataset.All.NACA.Airfoils.xlsx)

## How to Run MLP Model Training
```bash
python run train_mlp_aero.py
```

## How to Run Web Interface
```bash
python -m streamlit run streamlit_app.py


