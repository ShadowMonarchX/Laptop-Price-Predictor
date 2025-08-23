# ------------------ Import Libraries ------------------
import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# ------------------ Constants ------------------
PIPE_PATH = "pipe.pkl"
DF_PATH = "df.pkl"

# ------------------ Laptop Data ------------------
laptop_data = {
    "Brand": ["Lenovo", "HP", "Dell", "Apple", "Asus", "Samsung", "MI", "Microsoft", "MSI", "Acer"],
    "Models": [
        ["ThinkPad", "Legion", "Yoga", "IdeaPad"],
        ["Spectre x360", "Omen", "EliteBook", "Pavilion"],
        ["XPS", "Alienware", "Precision", "Inspiron"],
        ["MacBook Pro", "MacBook Air"],
        ["ZenBook", "ROG", "TUF"],
        ["Galaxy Book", "Odyssey"],
        ["Mi Notebook"],
        ["Surface Laptop", "Surface Pro"],
        ["Creator Series", "Stealth"],
        ["Predator", "Swift"]
    ],
    "UseCases": [
        ["Business", "AI Training", "Gaming", "Video Editing"],
        ["Business", "Gaming", "Video Editing"],
        ["Business", "Gaming", "AI Training", "Video Editing"],
        ["AI Training", "Video Editing", "Everyday"],
        ["Gaming", "AI Training", "Video Editing"],
        ["Everyday", "Gaming"],
        ["Everyday"],
        ["Business", "AI Training"],
        ["Gaming", "Video Editing"],
        ["Gaming", "Everyday"]
    ]
}

# Convert to DataFrame
df_laptops = pd.DataFrame(laptop_data)

# ------------------ App Class ------------------
class LaptopPricePredictor:
    def __init__(self):
        st.set_page_config(page_title="Laptop Price Predictor 💻", page_icon="💻", layout="wide")
        st.title("Laptop Price Predictor 💻")
        self.pipe = None
        self.df = None
        self.load_model_data()
        if self.pipe is None or self.df is None:
            self.train_demo_model()
        self.user_input()

    # ------------------ Load Model ------------------
    def load_model_data(self):
        try:
            if os.path.exists(PIPE_PATH) and os.path.exists(DF_PATH):
                self.pipe = joblib.load(PIPE_PATH)
                self.df = joblib.load(DF_PATH)
            else:
                st.warning("Model/data not found. Training demo model...")
        except Exception as e:
            st.error(f"Error loading model/data: {e}\nTraining demo model instead.")
            self.pipe, self.df = None, None

    # ------------------ Train Demo Model ------------------
    def train_demo_model(self):
        try:
            st.info("Training demo model...")

            # Dummy dataframe for pipeline
            self.df = pd.DataFrame({
                "Company": ["Dell", "HP", "Lenovo", "Apple", "Asus"],
                "TypeName": ["Ultrabook", "Notebook", "Gaming", "MacBook Pro", "ROG"],
                "Ram": [4, 8, 16, 16, 32],
                "Cpu brand": ["Intel", "AMD", "Intel", "Apple", "Intel"],
                "Gpu brand": ["Intel", "Nvidia", "Nvidia", "Apple", "Nvidia"],
                "os": ["Windows", "Windows", "Windows", "MacOS", "Windows"]
            })

            step1 = ColumnTransformer(transformers=[
                ("col_tnf", OneHotEncoder(handle_unknown='ignore'), 
                 ["Company", "TypeName", "Cpu brand", "Gpu brand", "os"])
            ], remainder="passthrough")

            estimators = [
                ("rf", RandomForestRegressor(n_estimators=10, random_state=3)),
                ("gbdt", GradientBoostingRegressor(n_estimators=10)),
            ]
            step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1), cv=2)

            self.pipe = Pipeline([("preprocessor", step1), ("regressor", step2)])

            # Dummy training data
            X_dummy = pd.DataFrame({
                "Company": ["Dell", "HP", "Lenovo", "Apple", "Asus"],
                "TypeName": ["Ultrabook", "Notebook", "Gaming", "MacBook Pro", "ROG"],
                "Ram": [4, 8, 16, 16, 32],
                "Cpu brand": ["Intel", "AMD", "Intel", "Apple", "Intel"],
                "Gpu brand": ["Intel", "Nvidia", "Nvidia", "Apple", "Nvidia"],
                "os": ["Windows", "Windows", "Windows", "MacOS", "Windows"],
                "Weight": [1.5, 2.0, 2.5, 1.4, 2.2],
                "Touchscreen": [0, 1, 0, 0, 1],
                "IPS": [1, 0, 1, 1, 1],
                "PPI": [141, 157, 160, 220, 144],
                "HDD": [0, 256, 512, 0, 512],
                "SSD": [256, 512, 1024, 512, 1024]
            })
            y_dummy = np.array([50000, 60000, 70000, 120000, 90000])

            self.pipe.fit(X_dummy, y_dummy)
            joblib.dump(self.pipe, PIPE_PATH)
            joblib.dump(self.df, DF_PATH)
            st.success("Demo model trained and saved!")
        except Exception as e:
            st.error(f"Failed to train demo model: {e}")

    # ------------------ User Input ------------------
    def user_input(self):
        st.write("### Enter your laptop specs:")

        # Brand & Model
        self.brand = st.selectbox("Select Brand", df_laptops["Brand"])
        models = df_laptops[df_laptops["Brand"] == self.brand]["Models"].values[0]
        self.model = st.selectbox("Select Model", models)

        # Use case
        use_cases = df_laptops[df_laptops["Brand"] == self.brand]["UseCases"].values[0]
        self.use_case = st.selectbox("Select Use Case", use_cases)

        # RAM, Weight, Screen
        self.ram = st.number_input("RAM (GB)", min_value=4, max_value=128, value=16, step=4)
        self.weight = st.number_input("Weight (kg)", 0.5, 5.0, 1.8)
        self.screen_size = st.number_input("Screen Size (inches)", 11.0, 20.0, 15.6)
        self.resolution = st.selectbox("Screen Resolution", ['1920x1080','1366x768','1600x900','3840x2160','2560x1600'])

        # CPU, GPU, Touchscreen
        self.cpu = st.selectbox("CPU Brand", ["Intel", "AMD", "Apple", "M1"])
        gpu_options = ["Intel", "Nvidia", "Apple", "AMD"]
        self.gpu = st.selectbox("GPU Brand", gpu_options)
        self.touchscreen = st.selectbox("Touchscreen", ["No","Yes"])
        self.ips = st.selectbox("IPS Display", ["No","Yes"])

        # Storage
        self.hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024])
        self.ssd = st.selectbox("SSD (GB)", [128, 256, 512, 1024, 2048])

        # OS selection (conditional logic)
        if self.brand == "Apple":
            self.os_type = st.selectbox("Operating System", ["MacOS", "iOS"])
        elif self.brand in ["Lenovo","HP","Dell","Asus","Samsung","MI","Microsoft","MSI","Acer"]:
            self.os_type = st.selectbox("Operating System", ["Windows", "Ubuntu", "Red Hat", "Kali Linux"])
        else:
            self.os_type = st.selectbox("Operating System", ["Windows", "Ubuntu"])

        # Predict button
        if st.button("Predict Price"):
            self.predict_price()

    # ------------------ Prediction ------------------
    def predict_price(self):
        try:
            touchscreen_val = 1 if self.touchscreen=="Yes" else 0
            ips_val = 1 if self.ips=="Yes" else 0
            X_res, Y_res = map(int, self.resolution.split("x"))
            ppi = ((X_res**2 + Y_res**2)**0.5) / self.screen_size

            query = pd.DataFrame([{
                "Company": self.brand,
                "TypeName": self.model,
                "Ram": self.ram,
                "Cpu brand": self.cpu,
                "Gpu brand": self.gpu,
                "os": self.os_type,
                "Weight": self.weight,
                "Touchscreen": touchscreen_val,
                "IPS": ips_val,
                "PPI": ppi,
                "HDD": self.hdd,
                "SSD": self.ssd
            }])
            predicted_price = int(self.pipe.predict(query)[0])
            st.success(f"💰 Predicted Laptop Price: Rs {predicted_price}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------ Run App ------------------
if __name__ == "__main__":
    LaptopPricePredictor()
