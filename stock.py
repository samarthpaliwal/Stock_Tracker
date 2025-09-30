import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.data = None
        self.features = None
        self.target = None

    def fetch_data(self, period='2y'):
        """Fetch historical stock data"""
        print(f"Fetching data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=period)
        print(f"Data fetched: {len(self.data)} days")
        return self.data

    def create_features(self):
        """Create technical indicators as features"""
        df = self.data.copy()

        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # Price momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()

        # Volume features
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Change'] = df['Volume'].pct_change()

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Price changes
        df['Daily_Return'] = df['Close'].pct_change()
        df['High_Low_Diff'] = df['High'] - df['Low']

        # Target: Next day's closing price
        df['Target'] = df['Close'].shift(-1)

        # Drop NaN values
        df = df.dropna()

        # Select features
        feature_columns = ['Close', 'Volume', 'MA_5', 'MA_10', 'MA_20', 'MA_50',
                           'Momentum_5', 'Momentum_10', 'Volatility',
                           'Volume_MA_5', 'Volume_Change', 'RSI',
                           'Daily_Return', 'High_Low_Diff']

        self.features = df[feature_columns]
        self.target = df['Target']

        return self.features, self.target

    def train_model(self, test_size=0.2):
        """Train the prediction model"""
        print("\nTraining model...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size, shuffle=False
        )

        self.model.fit(X_train, y_train)

        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print(f"\nModel Performance:")
        print(f"Train RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")

        return X_train, X_test, y_train, y_test, train_pred, test_pred

    def predict_next_day(self):
        """Predict next day's closing price"""
        latest_features = self.features.iloc[-1:].values
        prediction = self.model.predict(latest_features)[0]
        current_price = self.data['Close'].iloc[-1]
        change = prediction - current_price
        change_pct = (change / current_price) * 100

        print(f"\n{'='*50}")
        print(f"Prediction for {self.ticker}")
        print(f"{'='*50}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Next Close: ${prediction:.2f}")
        print(f"Expected Change: ${change:.2f} ({change_pct:+.2f}%)")
        print(f"{'='*50}")

        return prediction

    def plot_predictions(self, X_test, y_test, test_pred):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(15, 6))

        # Plot test data
        test_dates = self.data.index[-len(y_test):]
        plt.plot(test_dates, y_test.values,
                 label='Actual Price', color='blue', linewidth=2)
        plt.plot(test_dates, test_pred, label='Predicted Price',
                 color='red', linewidth=2, alpha=0.7)

        plt.title(
            f'{self.ticker} - Actual vs Predicted Stock Prices', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def feature_importance(self):
        """Display feature importance"""
        importance = pd.DataFrame({
            'Feature': self.features.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nFeature Importance:")
        print(importance.to_string(index=False))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance['Feature'], importance['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance in Stock Prediction')
        plt.tight_layout()
        plt.show()

        return importance


def main():
    # Example usage
    ticker = input("Enter stock ticker (e.g., AAPL, GOOGL, MSFT): ").upper()

    # Create predictor
    predictor = StockPredictor(ticker)

    # Fetch and prepare data
    predictor.fetch_data(period='2y')
    predictor.create_features()

    # Train model
    X_train, X_test, y_train, y_test, train_pred, test_pred = predictor.train_model()

    # Make next day prediction
    predictor.predict_next_day()

    # Show feature importance
    predictor.feature_importance()

    # Plot results
    predictor.plot_predictions(X_test, y_test, test_pred)


if __name__ == "__main__":
    main()
