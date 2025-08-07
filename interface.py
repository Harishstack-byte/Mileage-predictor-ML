import streamlit as st
import pandas as pd
import joblib

features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']

lr = joblib.load(r'C:\\Users\\haris\\OneDrive\\Desktop\\Mileage prediction model\\lrmodel.pkl')
rf = joblib.load(r'C:\\Users\\haris\\OneDrive\\Desktop\\Mileage prediction model\\rfmodel.pkl')

st.title("MPG Prediction App")
st.write("Enter vehicle details to predict mileage (MPG):")

cylinders = st.number_input("Cylinders", value=4)
displacement = st.number_input("Displacement", value=100)
horsepower = st.number_input("Horsepower", value=80)
weight = st.number_input("Weight", value=2500)
acceleration = st.number_input("Acceleration", value=10)
model_year = st.number_input("Model Year", value=80)

origin = st.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1: "USA", 2: "Europe", 3: "Japan"}[x])

if st.button("Predict Mileage"):
    user_input = pd.DataFrame([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]],
                              columns=features)

    
    lr_prediction = lr.predict(user_input)
    rf_prediction = rf.predict(user_input)

  
    st.subheader("Mileage Predictions:")
    st.write(f"Linear Regression: {lr_prediction[0]:.2f} MPG")
    st.write(f"Random Forest: {rf_prediction[0]:.2f} MPG")
