import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Traffic Prediction Dashboard", layout="wide")
st.title("🚦 Traffic Condition Prediction Dashboard")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    # Load only the RandomForest model
    model = joblib.load("rf_model.pkl")
    return model

model = load_model()

# Mapping for traffic condition
traffic_map = {'High': 2, 'Medium': 1, 'Low': 0}
reverse_traffic_map = {v: k for k, v in traffic_map.items()}

# -------------------------------
# SIDEBAR INPUTS (min/max from dataset)
# -------------------------------
st.sidebar.header("📥 Input Features")

def user_input():
    data = {
        'Latitude': st.sidebar.slider("Latitude", 40.600016, 40.899972, 40.749645),
        'Longitude': st.sidebar.slider("Longitude", -73.999987, -73.700159, -73.847433),
        'Vehicle_Count': st.sidebar.slider("Vehicle Count", 10, 299, 154),
        'Traffic_Speed_kmh': st.sidebar.slider("Traffic Speed (km/h)", 5.002789, 79.997556, 42.111096),
        'Road_Occupancy_%': st.sidebar.slider("Road Occupancy (%)", 10.005031, 99.999729, 54.748397),
        'Traffic_Light_State': st.sidebar.selectbox("Traffic Light State", [0,1,2]),
        'Weather_Condition': st.sidebar.selectbox("Weather Condition", [0,1,2,3,4]),
        'Accident_Report': st.sidebar.selectbox("Accident Report", [0,1]),
        'Sentiment_Score': st.sidebar.slider("Sentiment Score", -0.999819, 0.999354, -0.005652),
        'Ride_Sharing_Demand': st.sidebar.slider("Ride Sharing Demand", 1, 99, 50),
        'Parking_Availability': st.sidebar.slider("Parking Availability", 0, 49, 24),
        'Emission_Levels_g_km': st.sidebar.slider("Emission Levels (g/km)", 50.136855, 499.922663, 272.174927),
        'Energy_Consumption_L_h': st.sidebar.slider("Energy Consumption (L/h)", 5.003787, 29.995416, 17.343243),
        'hour': st.sidebar.slider("Hour", 0, 23, 11),
        'day': st.sidebar.slider("Day", 1, 18, 9)
    }
    return pd.DataFrame([data])

input_df = user_input()

# -------------------------------
# PREDICTION
# -------------------------------
st.subheader("🔮 Prediction")

pred_num = model.predict(input_df)[0]  # extract scalar
pred_label = reverse_traffic_map[pred_num]
probs = model.predict_proba(input_df)[0]

# -------------------------------
# KPI METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("🚗 Traffic Condition", pred_label)
col2.metric("📊 Confidence", f"{max(probs)*100:.2f}%")
col3.metric("🔢 Class", int(pred_num))

# -------------------------------
# PROBABILITY CHART
# -------------------------------
st.subheader("📊 Prediction Probability")

prob_df = pd.DataFrame({
    "Condition": [reverse_traffic_map[i] for i in range(len(probs))],
    "Probability": probs
})

fig = px.bar(prob_df, x="Condition", y="Probability",
             color="Condition", title="Prediction Probabilities")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.subheader("📈 Feature Importance")

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": input_df.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig2 = px.bar(feat_df, x="Importance", y="Feature", orientation='h',
              title="Feature Importance")
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# BATCH PREDICTION (CSV upload)
# -------------------------------
st.subheader("📂 Batch Prediction")

file = st.file_uploader("Upload CSV with same columns", type=["csv"])

if file:
    df = pd.read_csv(file)
    
    # Predict
    preds = model.predict(df)
    labels = [reverse_traffic_map[int(p)] for p in preds]
    
    df["Prediction"] = labels
    
    st.write(df.head())
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv")

# -------------------------------
# CONFUSION MATRIX DEMO (Optional)
# -------------------------------
st.subheader("📉 Confusion Matrix Demo")

if st.checkbox("Show confusion matrix example"):
    from sklearn.metrics import confusion_matrix
    
    # Demo data (replace with real test labels if available)
    y_true = np.random.randint(0,3,100)
    y_pred = np.random.randint(0,3,100)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig3, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig3)