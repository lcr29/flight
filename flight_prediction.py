import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

model = joblib.load('flight_prediction.pkl.gz')

# The list of columns from the training data
train_cols = [
    'Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min',
    'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins',
    'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
    'Airline_Jet Airways', 'Airline_Jet Airways Business',
    'Airline_Multiple carriers', 'Airline_Multiple carriers Premium economy',
    'Airline_SpiceJet', 'Airline_Trujet', 'Airline_Vistara',
    'Airline_Vistara Premium economy', 'Source_Chennai', 'Source_Delhi',
    'Source_Kolkata', 'Source_Mumbai', 'Destination_Cochin',
    'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',
    'Destination_New Delhi'
]

# Define a function to preprocess inputs
def process_input(airline, source, destination, total_stops, date_of_journey, dep_time, arrival_time, duration_hours, duration_mins):
    # Create a data frame
    df = pd.DataFrame({
        'Airline': [airline],
        'Source': [source],
        'Destination': [destination],
        'Total_Stops': [total_stops],
        'Date_of_Journey': [date_of_journey],
        'Dep_Time': [dep_time],
        'Arrival_Time': [arrival_time],
        'Duration_hours': [duration_hours],
        'Duration_mins': [duration_mins]
    })

    # Convert date and times to datetime objects
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], format='%H:%M')
    df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'], format='%H:%M')

    # Extract day, month, hour, and minute from datetime objects
    df['Journey_day'] = df['Date_of_Journey'].dt.day
    df['Journey_month'] = df['Date_of_Journey'].dt.month
    df['Dep_hour'] = df['Dep_Time'].dt.hour
    df['Dep_min'] = df['Dep_Time'].dt.minute
    df['Arrival_hour'] = df['Arrival_Time'].dt.hour
    df['Arrival_min'] = df['Arrival_Time'].dt.minute

    # Drop the original date and time columns
    df.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time'], axis=1, inplace=True)

    # One-hot encode the categorical variables using pd.get_dummies
    df = pd.get_dummies(df, columns=['Airline', 'Source', 'Destination'], drop_first=False)

    # Add missing dummy columns with 0s
    missing_cols = set(train_cols) - set(df.columns)
    for c in missing_cols:
        df[c] = 0

    # Remove extra columns
    extra_cols = set(df.columns) - set(train_cols)
    df = df.drop(columns=extra_cols)

    # Ensure the columns are in the same order as during training
    df = df[train_cols]

    # Ensure the Total_Stops is integer
    df['Total_Stops'] = df['Total_Stops'].astype(int)

    return df

# Streamlit application headings
st.image('header_image.png', width=150) 
st.title('Flight Price Prediction App')
st.write('Please enter the details of the flight')

# Create input fields for the user to enter the data required for prediction
airline = st.selectbox('Airline', ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia', 'Vistara Premium economy', 'Jet Airways Business', 'Multiple carriers Premium economy', 'Trujet'])
source = st.selectbox('Source', ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
destination = st.selectbox('Destination', ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
total_stops = st.selectbox('Total Stops', [0, 1, 2, 3, 4])
date_of_journey = st.date_input('Date of Journey').strftime('%d/%m/%Y')
dep_time = st.time_input('Departure Time').strftime('%H:%M')
arrival_time = st.time_input('Arrival Time').strftime('%H:%M')
duration_hours = st.number_input('Duration Hours', min_value=0)
duration_mins = st.number_input('Duration Minutes', min_value=0)

# When 'Predict' is clicked, make the prediction and store it
if st.button('Predict'):
    # Preprocess input data in the same way as training data
    processed_data = process_input(airline, source, destination, total_stops, date_of_journey, dep_time, arrival_time, duration_hours, duration_mins)

    # Get the prediction
    prediction = model.predict(processed_data)

    # Display the prediction
    st.success(f'The predicted flight price is ₹{round(prediction[0],2)}')
