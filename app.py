import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import datetime

# --- Load the model ---
model_path = r"C:\Users\sid24\OneDrive\Desktop\Machine learning\Future_Bitcoin_Prediction\Bitcoin_Price_prediction_Model.keras"
model = load_model(model_path)

# --- Streamlit UI ---
st.header('📈 Bitcoin Price Prediction Model')
st.subheader('Historical Bitcoin Price Data')

# Download historical Bitcoin data
data = yf.download('BTC-USD', start='2015-01-01', end=datetime.date.today())

data = data.reset_index()
st.write(data)

# Line chart of Bitcoin Close Price
st.subheader('Bitcoin Closing Price Line Chart')
st.line_chart(data['Close'])

# Prepare data
close_data = data[['Close']]
train_data = close_data[:-100]
test_data = close_data[-200:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

base_days = 100

# Prepare test inputs (x) and outputs (y)
x_test = []
y_test = []

for i in range(base_days, len(test_scaled)):
    x_test.append(test_scaled[i - base_days:i, 0])
    y_test.append(test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict on test data
st.subheader('🔁 Predicted vs Original Prices')
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
originals = scaler.inverse_transform(y_test.reshape(-1, 1))

comparison_df = pd.DataFrame({
    'Date': data['Date'][-len(predictions):].reset_index(drop=True),
    'Predicted Price': predictions.flatten(),
    'Original Price': originals.flatten()
})

st.write(comparison_df)
st.line_chart(comparison_df[['Predicted Price', 'Original Price']])

# Predict next 5 days
st.subheader('📅 Predicted Bitcoin Prices for Next 5 Days')

future_input = test_scaled[-base_days:, 0]
future_predictions = []

for _ in range(5):
    input_reshaped = future_input.reshape(1, base_days, 1)
    next_pred = model.predict(input_reshaped)[0, 0]
    future_predictions.append(next_pred)
    future_input = np.append(future_input[1:], next_pred)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=5)
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})

st.write(future_df)
st.line_chart(future_df.set_index('Date'))

# --- Calendar to Predict Price on Any Future Date ---
st.subheader("📆 Predict Bitcoin Price for Any Future Date (e.g., 2025)")

# Select any date in the future (up to end of 2025)
last_known_date = data['Date'].iloc[-1].date()
selected_future_date = st.date_input(
    "Select a future date",
    min_value=last_known_date + pd.Timedelta(days=1),
    max_value=datetime.date.today() + datetime.timedelta(days=365)

)

days_ahead = (selected_future_date - last_known_date).days

if days_ahead > 0:
    # Recursive prediction for the given number of days ahead
    future_input = test_scaled[-base_days:, 0].copy()
    predicted_scaled = []

    for _ in range(days_ahead):
        input_reshaped = future_input.reshape(1, base_days, 1)
        next_pred = model.predict(input_reshaped)[0, 0]
        predicted_scaled.append(next_pred)
        future_input = np.append(future_input[1:], next_pred)

    final_prediction = scaler.inverse_transform(np.array([[predicted_scaled[-1]]]))[0, 0]
    st.success(f"📈 Predicted Bitcoin price on {selected_future_date} is **${final_prediction:.2f}**")
else:
    st.warning("⚠️ Please select a future date.")
