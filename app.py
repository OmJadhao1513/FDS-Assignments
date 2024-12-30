import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import streamlit as st
import numpy as np
import joblib

# Streamlit App
st.title("Energy Consumption Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload your cleaned energy dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load and process the uploaded dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset successfully loaded!")

    # Display the first few rows
    st.write("Preview of the Dataset:")
    st.write(data.head())

    # Check for missing values
    st.write("Missing Values in Dataset:")
    st.write(data.isnull().sum())

    # Visualization: Pairplot
    st.write("Data Pairplot:")
    sns.pairplot(data)
    st.pyplot(plt)

    # Correlation Heatmap
    st.write("Correlation Heatmap:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

    # Normalize continuous variables
    st.write("Normalizing continuous variables...")
    scaler = StandardScaler()
    data[['Temperature (°C)', 'Humidity (%)']] = scaler.fit_transform(
        data[['Temperature (°C)', 'Humidity (%)']]
    )

    # Define features and target
    X = data[['Temperature (°C)', 'Humidity (%)', 'Occupancy']]
    y = data['Energy Consumption (kWh)']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    st.write("Training Ridge Regression Model...")
    model = Ridge()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model Mean Squared Error: {mse:.2f}")

    # Prediction Section
    st.subheader("Make a Prediction")

    temperature = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    occupancy = st.radio("Occupancy", (0, 1))

    if st.button("Predict"):
        input_features = scaler.transform([[temperature, humidity]])  # Apply scaling
        input_features = np.hstack([input_features, [[occupancy]]])  # Add occupancy
        prediction = model.predict(input_features)[0]
        st.write(f"Predicted Energy Consumption: {prediction:.2f} kWh")

    # Save and download the model
    joblib.dump(model, 'ridge_model.pkl')
    with open('ridge_model.pkl', 'rb') as file:
        st.download_button(
            label="Download Trained Model",
            data=file,
            file_name='ridge_model.pkl',
            mime='application/octet-stream'
        )
