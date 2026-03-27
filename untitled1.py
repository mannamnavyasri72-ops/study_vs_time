import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="Exam Score Predictor", layout="wide")
st.title("📚 Exam Score Predictor")
st.markdown("Predict exam scores based on study hours and sleep hours using Linear Regression")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("study_vs_time_500_rows.csv")
    return df

df = load_data()

# Display dataset columns for reference
st.write("Columns in dataset:", df.columns)

# -------------------------------
# Train Linear Regression Model
# -------------------------------
# Adjust column names here if your dataset has different headers
X = df[["study_hours", "sleep_hours"]]
y = df["exam_score"]

@st.cache_resource
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(X, y)

# -------------------------------
# Model Info
# -------------------------------
st.subheader("📊 Model Information")
col1, col2 = st.columns(2)
with col1:
    st.metric("Study Hours Coefficient", f"{model.coef_[0]:.2f}")
with col2:
    st.metric("Sleep Hours Coefficient", f"{model.coef_[1]:.2f}")
st.info(f"Intercept: {model.intercept_:.2f}")

# Model Accuracy
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
st.metric("Model Accuracy (R²)", f"{r2:.3f}")

# -------------------------------
# User Input Sliders
# -------------------------------
st.subheader("🎛️ Enter Your Data")
col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Study Hours",
        float(df["study_hours"].min()),
        float(df["study_hours"].max()),
        5.0,
        0.5
    )

with col2:
    sleep_hours = st.slider(
        "Sleep Hours",
        float(df["sleep_hours"].min()),
        float(df["sleep_hours"].max()),
        7.0,
        0.5
    )

# -------------------------------
# Make Prediction
# -------------------------------
input_data = pd.DataFrame({"study_hours": [study_hours], "sleep_hours": [sleep_hours]})
predicted_score = model.predict(input_data)[0]

st.success(f"🎯 Predicted Exam Score: {predicted_score:.2f}")

# -------------------------------
# Dataset Visualization
# -------------------------------
st.subheader("📉 Data Visualization")
st.write("Study Hours vs Exam Score")
st.scatter_chart(df, x="study_hours", y="exam_score")
st.write("Sleep Hours vs Exam Score")
st.scatter_chart(df, x="sleep_hours", y="exam_score")

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("📋 Dataset Preview")
st.dataframe(df.head())
