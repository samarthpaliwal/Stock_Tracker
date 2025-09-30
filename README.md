# Stock Predictor

A machine learning-based stock price predictor that uses technical indicators and Random Forest algorithm to forecast next-day closing prices.

## Features

- Real-time Data Fetching: Automatically fetches historical stock data using Yahoo Finance API
- Technical Indicators: Calculates 14+ technical indicators including:
  - Moving Averages (5, 10, 20, 50-day)
  - RSI (Relative Strength Index)
  - Momentum Indicators
  - Volatility Measures
  - Volume Analysis
- Machine Learning: Uses Random Forest Regressor for predictions
- Performance Metrics: Provides RMSE and R² scores for model evaluation
- Visualizations: Interactive charts showing actual vs predicted prices
- Feature Importance: Displays which indicators matter most for predictions

## Requirements

- Python 3.8+
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/samarthpaliwal/stock-tracker.git
cd stock-tracker
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

Run the main script:
```bash
python stock_predictor.py
```

Enter a stock ticker when prompted (e.g., AAPL, GOOGL, MSFT, TSLA).

### Example Output:
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

## How It Works

1. Data Collection: Fetches 2 years of historical stock data
2. Feature Engineering: Creates technical indicators from raw price/volume data
3. Model Training: Trains Random Forest model on 80% of data
4. Validation: Tests on remaining 20% and displays accuracy metrics
5. Prediction: Forecasts next trading day's closing price
6. Visualization: Generates comparison charts

## Technical Indicators Used

| Indicator | Description |
|-----------|-------------|
| Moving Averages | 5, 10, 20, 50-day averages |
| Momentum | 5 and 10-day price momentum |
| RSI | Relative Strength Index (14-day) |
| Volatility | 10-day rolling standard deviation |
| Volume Metrics | Volume moving average and change rate |
| Daily Returns | Percentage price changes |
| High-Low Spread | Daily price range |

## Model Performance

The Random Forest model typically achieves:
- R² Score: 0.95-0.98 on test data
- RMSE: $2-5 depending on stock volatility

## Disclaimer

This tool is for educational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Past performance does not guarantee future results. Always conduct your own research and consult with financial advisors before making investment decisions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

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
