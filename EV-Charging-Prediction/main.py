# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st

# Load and preprocess data
df = pd.read_csv("ev_charging_patterns.csv")

# Check for missing values
if df.isnull().sum().any():
    df = df.dropna()

# Encode categorical variables
label_encoder = LabelEncoder()
for col in ['Vehicle Model', 'User Type', 'Charger Type', 'Time of Day', 'Day of Week']:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

# Drop unnecessary columns
drop_columns = [
    "User ID", "Charging Station ID", "Charging Station Location",
    "Temperature (°C)", "Vehicle Age (years)", 
    "Distance Driven (since last charge) (km)", 
    "Charging Start Time", "Charging End Time",
    "Charging Cost (USD)"
]
x = df.drop(columns=drop_columns, errors='ignore')
y = df["Charging Cost (USD)"]

# Convert to numpy arrays
X = x.values
Y = y.values

# Initial Linear Regression Model
model = LinearRegression()
model.fit(X, Y)
y_pred = model.predict(X)
residuals = Y - y_pred

# Identify and remove outliers based on gradients
gradients = 2 * residuals
threshold = np.percentile(np.abs(gradients), 95)  # Top 5% gradients as outliers
outliers = np.abs(gradients) > threshold
X_filtered = X[~outliers]
y_filtered = Y[~outliers]

# Split filtered data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=2)

# Refit model on filtered data
model_filtered = LinearRegression()
model_filtered.fit(X_train, y_train)

# Predictions on test data
y_test_pred = model_filtered.predict(X_test)

# Streamlit Application
def main():
    st.title('Electric Vehicle Charging Cost Prediction')
    
    with st.form("prediction_form"):
        st.subheader("Input Parameters")
        
        vehicle_model = st.selectbox("Vehicle Model", options=[0, 1, 2, 3, 4],
                                     format_func=lambda x: {
                                         0: "BMW i3",
                                         1: "Chery Bolt",
                                         2: "Hyundai Kona",
                                         3: "Nissan Leaf",
                                         4: "Tesla Model 3"
                                     }[x])
        
        col1, col2 = st.columns(2)
        with col1:
            battery_capacity = st.number_input("Battery Capacity (kWh)")#, value=50.00000000000)
            energy_consumed = st.number_input("Energy Consumed (kWh)")#, value=30.0000000)
            charging_duration = st.number_input("Charging Duration (hours)")#, value=2.000000000)
            charging_rate = st.number_input("Charging Rate (kW)")#, value=50.0000000000)
        with col2:
            time_of_day = st.selectbox("Time of Day", options=[0, 1, 2, 3],
                                       format_func=lambda x: {
                                           0: "Afternoon",
                                           1: "Evening",
                                           2: "Morning",
                                           3: "Night"
                                       }[x])
            day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6],
                                       format_func=lambda x: {
                                           0: "Friday",
                                           1: "Monday",
                                           2: "Saturday",
                                           3: "Sunday",
                                           4: "Thursday",
                                           5: "Tuesday",
                                           6: "Wednesday"
                                       }[x])
            soc_start = st.number_input("State of Charge (Start %)")#, value=20.00000000000)
            soc_end = st.number_input("State of Charge (End %)")#, value=80.000000000000)
        
        charger_type = st.selectbox("Charger Type", options=[0, 1, 2],
                                    format_func=lambda x: {
                                        0: "DC Fast Charger",
                                        1: "Level-1",
                                        2: "Level-2"
                                    }[x])
        user_type = st.selectbox("User Type", options=[0, 1, 2],
                                 format_func=lambda x: {
                                     0: "Casual Driver",
                                     1: "Commuter",
                                     2: "Long Distance Traveler"
                                 }[x])
        
        submit_button = st.form_submit_button("Predict Charging Cost")
        
        if submit_button:
            
            # For demonstration, we'll just show the input values
            st.success("Prediction Complete!")
            
            st.subheader("Input Summary:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"Vehicle Model: {vehicle_model}")
                st.write(f"Battery Capacity: {battery_capacity} kWh")
                st.write(f"Energy Consumed: {energy_consumed} kWh")
                st.write(f"Charging Duration: {charging_duration} hours")
                st.write(f"Charging Rate: {charging_rate} kW")
            
            with col2:
                st.write(f"Time of Day: {time_of_day}")
                st.write(f"Day of Week: {day_of_week}")
                st.write(f"Initial SoC: {soc_start}%")
                st.write(f"Final SoC: {soc_end}%")
                st.write(f"Charger Type: {charger_type}")
                st.write(f"User Type: {user_type}")
                
            input_data = np.array([[
                vehicle_model, battery_capacity, energy_consumed,
                charging_duration, charging_rate, time_of_day,
                day_of_week, soc_start, soc_end, charger_type, user_type
            ]])
            prediction = model_filtered.predict(input_data)
            st.success(f"Predicted Charging Cost: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()
