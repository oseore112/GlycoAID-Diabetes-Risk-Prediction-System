import streamlit as st  # Builds the website
import pandas as pd  # Handles tables of data
import numpy as np  # Works with numbers
import joblib  # Loads the trained AI model
import plotly.express as px  # Draws charts and gauges
import plotly.graph_objects as go


# PAGE CONFIG
# Sets the name, icon, and size of the app
st.set_page_config(
    page_title="GlycoAID – Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"  # Makes the page look wide and professional
)


# LOAD SAVED MODEL
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()


# LOAD DATA (FOR FEATURE NAMES)
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
# get 'feature' names and show sample data
data = pd.read_csv(DATA_URL)
X = data.drop("Outcome", axis=1)


# APP TITLE
# Displays the big GlycoAID title
st.markdown(
    "<h1 style='text-align:center;'>🩺 GlycoAID – Diabetes Risk Prediction System</h1>",
    unsafe_allow_html=True
)

# Displays the sub-title
st.markdown(
    "<p style='text-align:center;color:gray;'>AI-powered early diabetes screening System</p>",
    unsafe_allow_html=True
)

st.divider()


# SIDEBAR INPUTS
# Create Sidebar Sidebar = where doctors enter patient data
# Each box collects one health detail

st.sidebar.header("Patient Information")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose Level", 50, 300, 120)
bp = st.sidebar.number_input("Blood Pressure", 40, 200, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 10, 100, 30)

input_data = np.array(
    [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])


# PREDICTION
prediction = model.predict(input_data)[0]

# Probability (safe handling)
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(input_data)[0][1]
else:
    prob = None


# MAIN RESULT DISPLAY
col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.error("🚨 **High Risk of Diabetes Detected**")
    else:
        st.success("✅ **Low Risk of Diabetes**")

with col2:
    if prob is not None:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Diabetes Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "crimson" if prob > 0.5 else "green"}
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

st.divider()


# MODEL COMPARISON TABLE (STATIC INFO)
st.subheader("Model Performance Summary")

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": ["~76.62%", "~78.57%"],
    "Strength": [
        "Stable and reliable",
        "Easy to understand decisions"
    ]
})

st.dataframe(comparison_df, use_container_width=True)


# FEATURE IMPORTANCE (FIXED FOR ALL MODELS)
st.divider()
st.subheader("Feature Importance")

# Handle Pipeline or Direct Model
if hasattr(model, "named_steps"):
    core_model = model.named_steps["model"]
else:
    core_model = model

# Logistic Regression
if hasattr(core_model, "coef_"):
    importance = np.abs(core_model.coef_[0])

# Decision Tree
elif hasattr(core_model, "feature_importances_"):
    importance = core_model.feature_importances_

else:
    st.warning("Feature importance not available.")
    st.stop()

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Teal"
)

st.plotly_chart(fig, use_container_width=True)


# EXPLAINABLE AI SECTION
st.divider()
st.subheader("Why This Prediction Was Made")

top_features = importance_df.head(3)

for _, row in top_features.iterrows():
    st.write(f"• **{row['Feature']}** had a strong effect on the prediction.")

st.info(
    "The AI compares patient data with patterns learned from past medical records. "
    "Higher values in key health indicators increase predicted diabetes risk."
)

# DATASET PREVIEW

st.divider()
with st.expander("📊 View Dataset Sample"):
    st.dataframe(data.head(10), use_container_width=True)


# FOOTER
st.markdown(
    "<p style='text-align:center;color:gray;'>Built with ❤️ using Python, Scikit-learn & Streamlit</p>",
    unsafe_allow_html=True
)
