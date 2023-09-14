import yfinance as yf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import datetime
from art import text2art

today = datetime.date.today()

# Generate ASCII art as logo
ascii_art = text2art("FiAi")

print("============================================================")

# Print the generated ASCII art
print(ascii_art)
print("Simple price prediction artificial intelligence")
print("============================================================")
print("Created by: Corvus Codex")
print("Github: https://github.com/CorvusCodex/")
print("Licence : MIT License")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")
print("For Symbols use https://finance.yahoo.com/lookup/")
print("============================================================")

# Ask the user for the stock or currency symbol
symbol = input("Enter the symbol of the stock or currency: ")

# Download historical data
print("Downloading historical data for training...")
today = datetime.date.today()
data = yf.download(symbol, start='2010-07-17', end=today)
print("Downloaded.")
print("Training...")

# Prepare data for model and make predictions for high, low, and close prices
predictions = {}
for price_type in ['High', 'Low', 'Close']:
    temp_data = data.copy()
    temp_data['Prediction'] = temp_data[price_type].shift(-1)
    temp_data.dropna(inplace=True)
    X = np.array(temp_data.drop(['Prediction'], axis=1))
    Y = np.array(temp_data['Prediction'])

    # Split data into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Make predictions
    temp_data['Prediction'] = model.predict(np.array(temp_data.drop(['Prediction'], axis=1)))
    
    # Store predictions
    predictions[price_type] = temp_data['Prediction']

# Print the predicted price for tomorrow, 7 days, 30 days, 1 year and 10 years from now
print("Training complete.")
print("============================================================")

for i in [1, 7, 30, 365, 3650]:
    if len(data) > i:
        print(f"Predicted prices for {i} day(s) from now:")
        print(f"High: {predictions['High'].iloc[-i]}")
        print(f"Low: {predictions['Low'].iloc[-i]}")
        print(f"Close: {predictions['Close'].iloc[-i]}")
        print("============================================================")
        
print("If you love this program, buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

# Prevent the window from closing immediately
input('Press ENTER to exit')
