
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Preprocessor (Fixed)
# -----------------------------
@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

@st.cache_resource
def load_model():
    return joblib.load("/Users/affanqureshi/Desktop/affan/Amazon Delivery Time Prediction/best_xgb_model.pkl")

try:
    preprocessor = load_preprocessor()
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model or preprocessor: {e}")
    st.stop()


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📦 Amazon Delivery Time Prediction App")
st.write("Enter delivery details below to predict estimated delivery time:")

# -----------------------------
# Input Categories (fixed or read from preprocessor)
# -----------------------------
categories = {
    "traffic": ["Low", "Medium", "High", "Jam"],
    "area": ["Urban", "Metropolitian", "Semi-Urban", "Other"],
    "weather": ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"],
    "vehicle": ['motorcycle', 'scooter', 'van'],
    "category": ["Electronics", "Clothing", "Groceries", "Other"]
}

# -----------------------------
# User Inputs
# -----------------------------
traffic = st.selectbox("Traffic", categories["traffic"])
area = st.selectbox("Area", categories["area"])
weather = st.selectbox("Weather", categories["weather"])
vehicle = st.selectbox("Vehicle", categories["vehicle"])
category = st.selectbox("Product Category", categories["category"])

distance = st.number_input("Delivery Distance (km)", min_value=0.0, step=0.1)
prep_time = st.number_input("Preparation Time (minutes)", min_value=0, step=1)
agent_age = st.number_input("Agent Age", min_value=18, max_value=70, step=1)
agent_rating = st.number_input("Agent Rating", min_value=1.0, max_value=5.0, step=0.1)

# -----------------------------
# Prepare Input DataFrame
# -----------------------------
input_df = pd.DataFrame([{
    "traffic": traffic,
    "area": area,
    "weather": weather,
    "vehicle": vehicle,
    "category": category,
    "delivery_distance_km": distance,
    "prep_time": prep_time,
    "store_latitude": 0,    # placeholders
    "store_longitude": 0,
    "drop_latitude": 0,
    "drop_longitude": 0,
    "agent_age": agent_age,
    "agent_rating": agent_rating
}])

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Delivery Time"):
    try:
        # Apply preprocessing
        X_transformed = preprocessor.transform(input_df)

        # Predict using the model
        prediction = model.predict(X_transformed)

        st.success(f"⏰ Estimated Delivery Time: {prediction[0]:.2f} minutes")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.info(
            "Make sure your preprocessor is fitted and all expected columns are provided. "
            "Recheck your preprocessor.pkl and model feature order."
        )
