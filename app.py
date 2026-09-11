import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import keras

# Page Config
st.set_page_config(page_title="Mobile Price Prediction", layout="wide")

st.markdown(
    """
<style>
header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 1001 !important;
}

/* 2. Container spacing fix */
.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 4.5rem !important;
}

/* 3. Smooth Hardware-Accelerated Header */
.custom-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 56px;
    background: #0e1117; 
    border-bottom: 1px solid #1f293d;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem;
    z-index: 999;
    box-sizing: border-box;
    will-change: transform;
}

.header-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc;
}

.header-tag:hover {
    transform: translateY(-1px);
}

/* 4. Smooth Hardware-Accelerated Footer */
.custom-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 44px;
    background: #0e1117;
    border-top: 1px solid #1f293d;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem;
    z-index: 999;
    font-size: 0.8rem;
    color: #94a3b8;
    box-sizing: border-box;
}

.footer-links a {
    color: #818cf8;
    text-decoration: none;
    margin-left: 14px;
    transition: color 0.15s ease;
}

.footer-links a:hover {
    color: #c7d2fe;
}

@media (max-width: 768px) {
    .custom-header {
        height: 50px;
        padding: 0 1rem;
    }
    .custom-footer {
        height: 40px;
        padding: 0 1rem;
        font-size: 0.72rem;
    }
    .header-brand {
        font-size: 0.95rem;
    }
    .header-tag {
        margin-right: 65px;
        font-size: 0.68rem;
        padding: 2px 8px;
    }
    .block-container {
        padding-top: 4rem !important;
        padding-bottom: 4rem !important;
    }
}
</style>

<!-- Header -->
<div class="custom-header">
    <div class="header-brand">
        <span>MobileAI</span>
    </div>
</div>

<!-- Footer -->
<div class="custom-footer">
    <div>© 2026 Mobile Price Prediction</div>
    <div class="footer-links">
        <a href="#about">About</a>
        <a href="#docs">Docs</a>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# 1. Load Model and Scaler
@st.cache_resource
def load_assets():
    model = keras.models.load_model("model.keras")
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, scaler


try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"error: {e}")

st.title("Mobile Price Classification")
st.write("Predict mobile price range based on specifications.")

# 2. Input Fields in Form
with st.form("mobile_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        battery_power_mah = st.number_input(
            "Battery Power (mAh)",
            min_value=500,
            max_value=5000,
            value=None,
            placeholder="e.g. 1000",
            step=1,
        )
        bluetooth = st.selectbox(
            "Bluetooth",
            options=["Yes", "No"],
            index=None,
            placeholder="Support Bluetooth?",
        )
        speed_of_microprocessor = st.slider(
            "Clock Speed (GHz)", min_value=0.5, max_value=3.0, step=0.1, value=1.5
        )
        dual_sim = st.selectbox(
            "Dual Sim",
            options=["Yes", "No"],
            index=None,
            placeholder="Support Dual SIM?",
        )
        front_camera = st.number_input(
            "Front Camera (Pixels/MP)",
            min_value=0,
            max_value=20,
            value=None,
            placeholder="e.g. 5",
            step=1,
        )
        fourG = st.selectbox(
            "4G", options=["Yes", "No"], index=None, placeholder="Support 4G?"
        )
        internal_memory = st.number_input(
            "Internal Memory (GB)",
            min_value=2,
            max_value=512,
            value=None,
            placeholder="e.g. 32",
            step=1,
        )

    with col2:
        mobile_depth = st.slider(
            "Mobile Depth (cm)", min_value=0.1, max_value=1.5, step=0.1, value=0.5
        )
        mobile_weight = st.number_input(
            "Mobile Weight (g)",
            min_value=50,
            max_value=300,
            value=None,
            placeholder="e.g. 140",
            step=1,
        )
        cores_of_processor = st.slider(
            "Cores of Processor", min_value=2, max_value=8, step=2, value=4
        )
        primary_camera = st.number_input(
            "Primary Camera (Pixels/MP)",
            min_value=0,
            max_value=30,
            value=None,
            placeholder="e.g. 12",
            step=1,
        )
        px_height = st.number_input(
            "Pixel Height (ppcm)",
            min_value=0,
            max_value=2000,
            value=None,
            placeholder="e.g. 1000",
            step=1,
        )
        px_width = st.number_input(
            "Pixel Width (ppcm)",
            min_value=500,
            max_value=2500,
            value=None,
            placeholder="e.g. 1500",
            step=1,
        )
        ram_mb = st.number_input(
            "RAM (MB)",
            min_value=256,
            max_value=6000,
            value=None,
            placeholder="e.g. 2048",
            step=1,
        )

    with col3:
        screen_height = st.number_input(
            "Screen Height (cm)",
            min_value=1,
            max_value=25,
            value=None,
            placeholder="e.g. 12",
            step=1,
        )
        screen_width = st.number_input(
            "Screen Width (cm)",
            min_value=1,
            max_value=20,
            value=None,
            placeholder="e.g. 5",
            step=1,
        )
        talk_time = st.slider(
            "Talk-Time (Hours)", min_value=2, max_value=20, step=1, value=10
        )
        threeG = st.selectbox(
            "3G", options=["Yes", "No"], index=None, placeholder="Support 3G?"
        )
        touch_screen = st.selectbox(
            "Touch Screen",
            options=["Yes", "No"],
            index=None,
            placeholder="Has Touch Screen?",
        )
        wifi = st.selectbox(
            "Wifi", options=["Yes", "No"], index=None, placeholder="Has Wifi?"
        )

    submitted = st.form_submit_button("Predict Price Category")

# 3. Prediction Pipeline
if submitted:
    # check value is null or not
    inputs = [
        battery_power_mah,
        bluetooth,
        dual_sim,
        front_camera,
        fourG,
        internal_memory,
        mobile_weight,
        primary_camera,
        px_height,
        px_width,
        ram_mb,
        screen_height,
        screen_width,
        threeG,
        touch_screen,
        wifi,
    ]

    if None in inputs:
        st.warning("Fill  all the blank fields first.")
    else:
        input_data = pd.DataFrame(
            [
                {
                    "Battery_power_mAh": battery_power_mah,
                    "Bluetooh": bluetooth,
                    "Speed_of_microprocessor": speed_of_microprocessor,
                    "Dual_sim": dual_sim,
                    "Front_camera": front_camera,
                    "4G": fourG,
                    "Internal_memeory_gb": internal_memory,
                    "Mobile_depth": mobile_depth,
                    "Mobile_weight": mobile_weight,
                    "Cores_of_processor": cores_of_processor,
                    "Primary_camera": primary_camera,
                    "px_height": px_height,
                    "Pixel_width": px_width,
                    "Ram_mb": ram_mb,
                    "Screen_height": screen_height,
                    "Screen_weight": screen_width,
                    "talk_time": talk_time,
                    "3G": threeG,
                    "touch_screen": touch_screen,
                    "wifi": wifi,
                }
            ]
        )

        # Binary mapping
        cols_to_map = ["Bluetooh", "Dual_sim", "4G", "3G", "touch_screen", "wifi"]
        mapping = {"Yes": 1, "No": 0}
        input_data[cols_to_map] = input_data[cols_to_map].replace(mapping)

        # Ensure correct datatypes
        input_data = input_data.astype(float)

        # Scale features
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        price_classes = ["High Cost", "Low Cost", "Medium Cost", "Very High Cost"]

        st.success(
            f"### Predicted Price Range: **{price_classes[predicted_class_index]}**"
        )
