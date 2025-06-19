
# 📈 Bitcoin Price Predictor

A **machine learning-powered web application** to predict future Bitcoin prices using historical data. Built with **Keras (LSTM)** and **Streamlit**, this app lets users view historical trends, compare real vs predicted values, and forecast prices for future dates.

---

## 🔧 Features

- 📊 Visualize historical Bitcoin price data (2015 to present)
- 🤖 Predict prices using a trained LSTM model
- 🔁 Show actual vs predicted prices for recent data
- 📅 Forecast next 5 days of Bitcoin prices
- 🗓️ Use an interactive calendar to forecast price for any date up to a year ahead

---

## 🧠 Model

- Model Type: LSTM (Long Short-Term Memory Neural Network)
- Framework: Keras
- Input: Past 100 days of Bitcoin closing prices
- Output: Next day predicted closing price
- File: `Bitcoin_Price_prediction_Model.keras`

---

## 🗂️ Folder Structure

```
Bitcoin_Predictor/
│
├── app.py                             # Streamlit application file
├── Bitcoin_Price_prediction_Model.keras  # Trained LSTM model
├── requirements.txt                   # Project dependencies
├── Bitcoin_predictor.ipynb            # Jupyter Notebook (training/testing)
└── README.md                          # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bitcoin-predictor.git
cd bitcoin-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📦 Dependencies

- streamlit
- yfinance
- pandas
- numpy
- keras
- scikit-learn

---

## 📝 Usage

- View and explore historical price charts.
- Compare model predictions vs actual prices.
- Predict prices for the next 5 days automatically.
- Use the calendar to select any future date and get the predicted value.

---

## 📌 Notes

- This app dynamically pulls the latest Bitcoin data using Yahoo Finance.
- Forecasts are generated based on recursive predictions from the trained model.

---

## 📬 Contact

For contributions or queries, contact [sid242294@gmail.com].
