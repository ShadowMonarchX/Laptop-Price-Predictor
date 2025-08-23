# Importing Libraries
import streamlit as st
import pickle
import numpy as np
import sys

st.set_page_config(
    page_title="Laptop Price Predictor 💻",
    page_icon="💻",
    layout="wide"
)

st.title("Laptop Price Predictor 💻")

# ------------------ Load Model & Data ------------------
try:
    pipe = pickle.load(open("pipe.pkl", "rb"))
    df = pickle.load(open("df.pkl", "rb"))
except Exception as e:
    st.error("Error loading model or data. Make sure 'pipe.pkl' and 'df.pkl' are created in this environment.")
    st.stop()

st.write("### Enter the specifications of your laptop:")

# ------------------ First Row ------------------
left_col, middle_col, right_col = st.columns(3)
with left_col:
    company = st.selectbox("Brand", df["Company"].unique())
with middle_col:
    laptop_type = st.selectbox("Type", df["TypeName"].unique())
with right_col:
    ram = st.selectbox("Ram (in GB)", df["Ram"].unique())

# ------------------ Second Row ------------------
left_col, middle_col, right_col = st.columns(3)
with left_col:
    weight = st.number_input("Weight of laptop (kg)", min_value=0.5, max_value=5.0, step=0.1)
with middle_col:
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
with right_col:
    ips = st.selectbox("IPS Display", ["No", "Yes"])

# ------------------ Third Row ------------------
left_col, middle_col, right_col = st.columns(3)
with left_col:
    screen_size = st.number_input("Screen Size (in Inches)", min_value=10.0, max_value=20.0, step=0.1)
with middle_col:
    resolution = st.selectbox(
        "Screen Resolution",
        ['1920x1080', '1366x768', '1600x900', '3840x2160',
         '3200x1800', '2880x1800', '2560x1600','2560x1440', '2304x1440']
    )
with right_col:
    cpu = st.selectbox("CPU Brand", df["Cpu brand"].unique())

# ------------------ Fourth Row ------------------
left_col, right_col = st.columns(2)
with left_col:
    hdd = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])
with right_col:
    ssd = st.selectbox("SSD (in GB)", [0, 8, 128, 256, 512, 1024])

# ------------------ Fifth Row ------------------
gpu = st.selectbox("GPU Brand", df["Gpu brand"].unique())
os = st.selectbox("OS Type", df["os"].unique())

# ------------------ Prediction ------------------
if st.button("Predict Price"):
    try:
        touchscreen_val = 1 if touchscreen == "Yes" else 0
        ips_val = 1 if ips == "Yes" else 0

        X_res, Y_res = map(int, resolution.split("x"))
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        query = np.array([
            company, laptop_type, ram, weight, touchscreen_val, ips_val,
            ppi, cpu, hdd, ssd, gpu, os
        ]).reshape(1, 12)

        predicted_price = int(np.exp(pipe.predict(query)[0]))
        st.success(f"💰 The Predicted Price of Laptop = Rs {predicted_price}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
