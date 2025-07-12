# src/mlops_iris/ui.py

import streamlit as st
from mlops_iris.api import get_predictions

def show_button() -> None:
    if st.button("Click me"):
        prediction = get_predictions()['predictions'][0]
        st.write(f"Prediction: {prediction}")