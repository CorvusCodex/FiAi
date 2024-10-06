import streamlit as st
import yfinance as yf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
from art import text2art

# Function to download historical data and make predictions
def predict_prices(symbol):
    today = datetime.date.today()
    data = yf.download(symbol, start='2010-07-17', end=today)
    predictions = {}

    for price_type in ['High', 'Low', 'Close']:
        temp_data = data.copy()
        temp_data['Prediction'] = temp_data[price_type].shift(-1)
        temp_data.dropna(inplace=True)
        X = np.array(temp_data.drop(['Prediction'], axis=1))
        Y = np.array(temp_data['Prediction'])

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        model = LinearRegression()
        model.fit(x_train, y_train)
        temp_data['Prediction'] = model.predict(np.array(temp_data.drop(['Prediction'], axis=1)))
        predictions[price_type] = temp_data['Prediction']

    return predictions, data

# Streamlit app
def main():
    st.set_page_config(
        page_title="Stock Market Prediction",
        page_icon=":chart_with_upwards_trend:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title(":chart_with_upwards_trend: Stock Market Prediction")
    st.image("stock.jpg")

    st.sidebar.title("User Input")
    symbol = st.sidebar.text_input("Enter the symbol of the stock (e.g., AAPL, META, etc)")

    if st.sidebar.button("Predict", key="predict_button"):
        if not symbol:
            st.warning("Please enter a symbol.")
        else:
            try:
                predictions, data = predict_prices(symbol.upper())

                st.subheader("Predicted Prices")
                for i in [1, 7, 30, 365, 3650]:
                    if len(data) > i:
                        with st.expander(f"{i} day(s) from now"):
                            st.write(f"- High: {predictions['High'].iloc[-i]:.2f}")
                            st.write(f"- Low: {predictions['Low'].iloc[-i]:.2f}")
                            st.write(f"- Close: {predictions['Close'].iloc[-i]:.2f}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
