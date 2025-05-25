import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# Load the model and preprocessors
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# Collect user inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.slider("Tenure (Years)", min_value=0, max_value=60, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", ['Yes', 'No'])
is_active_member = st.selectbox("Is Active Member", ['Yes', 'No'])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Map binary inputs to 0 and 1
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# Prediction logic
if st.button("Predict"):
    # Encode categorical variables
    gender_encoded = label_encoder_gender.transform([gender])[0]
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()[0]

    # Create input dataframe with correct column names and order
    input_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Create geography dataframe
    geo_df = pd.DataFrame([geo_encoded], columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Concatenate dataframes
    input_df = pd.concat([input_df, geo_df], axis=1)

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0][0]

    # Display results
    st.subheader("Prediction Result")
    if prediction > 0.5:
        st.write("**Prediction:** Churn")
    else:
        st.write("**Prediction:** Not Churn")
    st.write(f"**Probability:** {prediction:.2f}")