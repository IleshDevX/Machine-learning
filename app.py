import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

st.title("Lasso Regression Predictor")

# Dynamically set number of features
num_features = model.n_features_in_

st.header("Enter feature values:")
inputs = [st.number_input(f"Feature {i+1}", value=0.0, format="%.4f") for i in range(num_features)]

if st.button("Predict"):
    prediction = model.predict([inputs])
    st.success(f"Predicted Output: {prediction[0]:.4f}")
