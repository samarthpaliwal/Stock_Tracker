# Stock Predictor

A machine learning-based stock price predictor that uses technical indicators and a Random Forest algorithm to forecast next-day closing prices. Given a ticker symbol, it pulls two years of daily history from Yahoo Finance, derives moving averages, momentum, RSI, volatility and volume features, trains a `RandomForestRegressor` on the first 80% of the series, reports error metrics on the last 20%, and prints a prediction for the next trading day's close along with charts.

## Features

- Data fetching: downloads historical stock data through the `yfinance` wrapper around Yahoo Finance
- Technical indicators: 14 feature columns including
  - Moving averages (5, 10, 20, 50-day)
  - Momentum indicators (5 and 10-day)
  - Volatility (10-day rolling standard deviation)
  - Volume analysis (5-day volume average, volume change rate)
  - RSI (14-day), daily returns, high-low spread
- Machine learning: Random Forest Regressor (100 trees) for predictions
- Performance metrics: RMSE and R² on both training and test data
- Visualizations: chart of actual vs predicted prices over the test period
- Feature importance: table and bar chart of which indicators the model relied on most

## Requirements

- Python 3.8+
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib
- Internet access (Yahoo Finance is queried at runtime)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/samarthpaliwal/Stock_Tracker.git
   cd Stock_Tracker
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install yfinance pandas numpy scikit-learn matplotlib
   ```

## Usage

Run the main script (the file is `stock.py`):

```bash
python stock.py
```

Enter a stock ticker when prompted (e.g., AAPL, GOOGL, MSFT, TSLA). The script has no command-line flags; the ticker is the only input and the period is fixed at two years in `main()`.

What you see, in order:

1. Fetch progress and the number of trading days retrieved.
2. Model performance block with train and test RMSE (in dollars) and R².
3. Prediction block with the current close, predicted next close, and expected change.
4. Feature importance table in the console, then a bar chart window.
5. A line chart window comparing actual and predicted prices on the test set.

Each matplotlib window blocks until closed.

### Example output

```
Enter stock ticker (e.g., AAPL, GOOGL, MSFT): AAPL
Fetching data for AAPL...
Data fetched: 504 days

Training model...

Model Performance:
Train RMSE: $2.45
Test RMSE: $3.12
Train R²: 0.9856
Test R²: 0.9723

==================================================
Prediction for AAPL
==================================================
Current Price: $178.45
Predicted Next Close: $179.20
Expected Change: $0.75 (+0.42%)
==================================================
```

(Values are illustrative; actual numbers depend on the ticker and date.)

## How it works

1. Data collection: `StockPredictor.fetch_data` calls `yf.Ticker(ticker).history(period='2y')`.
2. Feature engineering: `create_features` builds the indicator columns from Close, High, Low and Volume, sets the target to the next day's Close (`shift(-1)`), and drops rows with NaNs from the rolling windows.
3. Model training: `train_model` splits chronologically (`shuffle=False`) into 80% train and 20% test, fits the Random Forest, and prints RMSE and R² for both splits.
4. Prediction: `predict_next_day` feeds the most recent feature row to the model and reports the change versus the latest close.
5. Reporting: `feature_importance` prints and plots `feature_importances_`; `plot_predictions` plots actual vs predicted over the test window.

Because the split is chronological and a tree model cannot extrapolate beyond the price range it was trained on, test metrics can look very different from the illustrative numbers above, especially for stocks that trended strongly in the last 20% of the window.

## Technical indicators used

| Indicator | Description |
|-----------|-------------|
| Moving Averages | 5, 10, 20, 50-day averages |
| Momentum | 5 and 10-day price momentum |
| RSI | Relative Strength Index (14-day) |
| Volatility | 10-day rolling standard deviation |
| Volume Metrics | Volume moving average and change rate |
| Daily Returns | Percentage price changes |
| High-Low Spread | Daily price range |

## Project layout

- `stock.py`: the `StockPredictor` class and the interactive `main()` entry point.
- `requirements.txt`: pinned minimum versions of the five dependencies.

## Disclaimer

This tool is for educational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Past performance does not guarantee future results. Always conduct your own research and consult with financial advisors before making investment decisions.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is intended to be released under the MIT License. No LICENSE file is currently included in the repository.

## Future enhancements

- Add LSTM/Neural Network models
- Support for multiple timeframe predictions
- Real-time streaming predictions
- Web interface with Flask/Django
- Portfolio optimization features
- Sentiment analysis from news/social media
- Backtesting framework
- Support for cryptocurrency predictions

## Acknowledgments

- yfinance - Yahoo Finance API wrapper
- scikit-learn - Machine learning library
- pandas - Data manipulation library
