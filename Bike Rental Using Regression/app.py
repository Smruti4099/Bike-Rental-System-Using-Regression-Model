import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load saved files
# -----------------------------
model = joblib.load("bike_demand_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🚲 Bike Rental Demand Prediction App")

st.write("Enter input values to predict bike rental demand")

# -----------------------------
# User Inputs
# -----------------------------
season = st.selectbox("Season", [1, 2, 3, 4])
yr = st.selectbox("Year (0 = 2018, 1 = 2019)", [0, 1])
mnth = st.slider("Month", 1, 12, 7)
hr = st.slider("Hour", 0, 23, 8)
holiday = st.selectbox("Holiday", [0, 1])
workingday = st.selectbox("Working Day", [0, 1])
weathersit = st.selectbox("Weather Situation", [1, 2, 3, 4])

temp = st.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temperature", 0.0, 1.0, 0.5)
hum = st.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.slider("Wind Speed", 0.0, 1.0, 0.2)

day = st.slider("Day of Month", 1, 31, 15)
dayofweek = st.slider("Day of Week (0=Sun)", 0, 6, 2)

# Peak hour logic (same as training)
is_peak_hour = 1 if (7 <= hr <= 9 or 17 <= hr <= 19) else 0

# -----------------------------
# Create input DataFrame
# -----------------------------
input_data = {
    "season": season,
    "yr": yr,
    "mnth": mnth,
    "hr": hr,
    "holiday": holiday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp": temp,
    "atemp": atemp,
    "hum": hum,
    "windspeed": windspeed,
    "day": day,
    "dayofweek": dayofweek,
    "is_peak_hour": is_peak_hour,
    "casual": 0,
    "registered": 0
}

input_df = pd.DataFrame([input_data])

# -----------------------------
# Apply same scaling
# -----------------------------
scale_cols = ['temp', 'atemp', 'hum', 'windspeed']
input_df[scale_cols] = scaler.transform(input_df[scale_cols])

# -----------------------------
# ALIGN FEATURES (CRITICAL FIX)
# -----------------------------
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.write("Prediction raw value:", prediction)
    st.success(f"🚴 Predicted Bike Rental Count: {round(prediction[0],2)}")