from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import streamlit as st

df = pd.read_csv("study_vs_time_500_rows.csv")

X = df[["study_hours", "sleep_hours"]]  
y = df["exam_score"]

model = LinearRegression()
model.fit(X, y)

y_pred=model.predict(X)
user_input1=float(input("Enter study_hours"))
user_input2=float(input("Enter sleep_hours"))
input_data=pd.DataFrame([{
    "study_hours":user_input1,
    "sleep_hours":user_input2
}])
print("prediction values:",model.predict(input_data))

st.title("Exam Score Predictor")

user_input1 = st.slider("Enter study hours:", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
user_input2 = st.slider("Enter sleep hours:", min_value=0.0, max_value=24.0, value=8.0, step=0.5)

if st.button("Predict"):
    predicted_score = model.predict([[user_input1, user_input2]])
    st.write(f"Predicted Exam Score: {predicted_score[0]:.2f}")
