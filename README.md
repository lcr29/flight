# Flight Price Prediction App
# Overview
This project is a Streamlit web application for predicting flight prices. It uses a machine learning model trained on various features like airlines, journey dates, times, and stops to estimate the price of flights.

#Features
Predictive model trained with a Random Forest algorithm.
Interactive Streamlit interface for easy use.
Pre-processing of input data to match the model's training format.
Visualization and user-friendly interface for entering flight details and viewing predictions.

# Input Fields
Airline: Select the airline from the dropdown.
Source: Choose the flight's starting location.
Destination: Select the flight's destination.
Total Stops: Choose the number of stops for the flight.
Date of Journey: Pick the date of the journey.
Departure Time: Set the flight's departure time.
Arrival Time: Set the flight's arrival time.
Duration Hours: Enter the flight's duration in hours.
Duration Minutes: Enter the flight's duration in minutes.
After filling in these details, click the 'Predict' button to view the flight price prediction.

# Model Details
The machine learning model was trained using a Random Forest Regressor. The model was optimized using RandomizedSearchCV for the best parameters. Key features considered include airlines, source and destination cities, journey dates, times, and flight duration.
