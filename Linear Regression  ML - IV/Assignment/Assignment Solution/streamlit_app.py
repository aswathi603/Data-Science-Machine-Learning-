import streamlit as st
import pandas as pd
from predict_housing_medv import predict_housing_medv

# Page setup
st.set_page_config(
    page_title="Boston Housing MEDV Predictor",
    layout="wide"
)

# Load model
predictor = predict_housing_medv()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- TITLE --------------------
st.title("🏠 Boston Housing MEDV Predictor")
st.markdown(
    "A **Machine Learning application** using **Linear Regression** to predict "
    "median house prices (MEDV)."
)

st.divider()

# -------------------- INPUTS --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Environmental Factors")
    crim = st.slider("CRIM – Crime rate", 0.00632, 88.9762, 0.00632)
    zn = st.slider("ZN – Residential land zoned (%)", 0.0, 100.0, 0.0)
    indus = st.slider("INDUS – Non-retail business acres", 0.46, 27.74, 0.46)
    chas = st.radio("CHAS – Near Charles River?", [0, 1])
    nox = st.slider("NOX – Nitric oxide concentration", 0.385, 0.871, 0.385)
    dis = st.slider("DIS – Distance to employment centers", 1.1296, 12.1265, 1.1296)

with col2:
    st.subheader("🏡 Housing & Social Factors")
    rm = st.slider("RM – Avg rooms per dwelling", 3.561, 8.78, 3.561)
    age = st.slider("AGE – Houses built before 1940 (%)", 2.9, 100.0, 2.9)
    rad = st.slider("RAD – Highway accessibility", 1.0, 24.0, 1.0)
    tax = st.slider("TAX – Property tax rate", 187.0, 711.0, 187.0)
    ptratio = st.slider("PTRATIO – Pupil-teacher ratio", 12.6, 22.0, 12.6)
    black = st.slider("BLACK – Proportion of blacks", 0.32, 396.9, 0.32)
    lstat = st.slider("LSTAT – Lower status population (%)", 1.73, 37.97, 1.73)

features = [
    crim, zn, indus, chas, nox, rm,
    age, dis, rad, tax, ptratio, black, lstat
]

st.divider()

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict MEDV", use_container_width=True):
    prediction = predictor.predict(features)

    # Save history
    st.session_state.history.append({
        "CRIM": crim,
        "RM": rm,
        "LSTAT": lstat,
        "MEDV ($k)": round(prediction, 2)
    })

    st.success(f"🏷️ **Predicted MEDV:** ${prediction:.2f} (in thousands)")

# -------------------- HISTORY --------------------
if st.session_state.history:
    st.subheader("📜 Prediction History")
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)

    if st.button("🧹 Clear History"):
        st.session_state.history.clear()

# -------------------- FEATURE IMPORTANCE --------------------
st.divider()
st.subheader("📈 Feature Importance (Model Insight)")

importance = predictor.get_feature_importance()
imp_df = pd.DataFrame({
    "Feature": importance.keys(),
    "Coefficient": importance.values()
}).sort_values(by="Coefficient", ascending=False)

st.bar_chart(imp_df.set_index("Feature"))

# -------------------- SIDEBAR --------------------
st.sidebar.title("ℹ️ Project Info")
st.sidebar.markdown("""
**Model:** Linear Regression  
**Scaler:** StandardScaler  
**Dataset:** Boston Housing  
**Target:** MEDV  

📌 This app demonstrates:
- ML pipeline
- Feature scaling
- Model inference
- UI deployment
""")

# -------------------- FOOTER --------------------
st.markdown("---")