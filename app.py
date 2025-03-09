import streamlit as st
import pickle
import numpy as np

# Load the trained model, scaler, and PCA transformer
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("pca.pkl", "rb") as file:
    pca = pickle.load(file)

# Title
st.title("Predictive Maintenance - Failure Prediction Model")

# Sidebar Inputs
st.sidebar.header("Enter Feature Values")

# Define feature names (as per dataset)
feature_names = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
    "Motor_current", "COMP", "DV_electric", "Towers", "MPG", "LPS",
    "Pressure_switch", "Oil_level", "Caudal_impulses"
]

# Get user inputs for all original features
user_inputs = []
for feature in feature_names:
    user_inputs.append(st.sidebar.number_input(f"{feature}", value=0.0))

# Convert inputs to NumPy array
input_data = np.array([user_inputs])

# Step 1: Scale Input Data
input_scaled = scaler.transform(input_data)

# Step 2: Apply PCA Transformation
input_pca = pca.transform(input_scaled)

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_pca)
    st.write("### Prediction:", prediction[0])
