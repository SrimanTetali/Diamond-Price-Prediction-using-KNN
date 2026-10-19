import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")

MODEL_PATH = Path(__file__).parent / "knn_diamond_pipeline.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("💎 Diamond Price Prediction")
st.write("Predict a diamond's price (USD) with a K-Nearest Neighbors regression model.")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat", min_value=0.2, max_value=5.01, value=0.5, step=0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    color = st.selectbox("Color (D = best, J = worst)", ["D", "E", "F", "G", "H", "I", "J"], index=3)
    clarity = st.selectbox(
        "Clarity (IF = best, I1 = worst)",
        ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
        index=3,
    )

with col2:
    depth = st.number_input("Depth %", min_value=43.0, max_value=79.0, value=61.8, step=0.1)
    table = st.number_input("Table %", min_value=43.0, max_value=95.0, value=57.0, step=0.1)
    x = st.number_input("Length x (mm)", min_value=0.1, max_value=11.0, value=5.0, step=0.01)
    y = st.number_input("Width y (mm)", min_value=0.1, max_value=60.0, value=5.0, step=0.01)
    z = st.number_input("Depth z (mm)", min_value=0.1, max_value=32.0, value=3.1, step=0.01)

if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame(
        [
            {
                "carat": carat,
                "cut": cut,
                "color": color,
                "clarity": clarity,
                "depth": depth,
                "table": table,
                "x": x,
                "y": y,
                "z": z,
            }
        ]
    )
    price = model.predict(input_df)[0]
    st.success(f"Estimated price: ${price:,.2f}")
