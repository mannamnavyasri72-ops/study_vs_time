
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv("/content/study_vs_time_500_rows.csv")
x = df[["study_hours", "sleep_hours"]]
y = df["exam_score"]

# Train the model
model = LinearRegression()
model.fit(x, y)

st.title("Exam Score Predictor")

st.write("Adjust the sliders below to predict your exam score based on study and sleep hours.")

# Create input widgets using sliders
# Based on the kernel state, study_hours are generally 0-5, sleep_hours 4-9.
# Setting reasonable min/max values for the sliders.
study_hours = st.slider("Study Hours", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sleep_hours = st.slider("Sleep Hours", min_value=4.0, max_value=10.0, value=7.0, step=0.1)

# Prepare input for prediction
user_data = pd.DataFrame([{
    "study_hours": study_hours,
    "sleep_hours": sleep_hours
}])

# Make prediction
predicted_score = model.predict(user_data)[0]

st.write(f"### Predicted Exam Score: {predicted_score:.2f}")