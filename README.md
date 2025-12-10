# stock-prediction-from-scratch

A simple, educational Flask application that demonstrates a reproducible pipeline for stock forecasting. This project shows how to download historical market data, create features, train models, evaluate them with time-series-aware methods, and serve predictions via a lightweight web app.

Goals
- Provide a clear, minimal end-to-end example so readers can learn by reading and extending.
- Keep code simple and well-documented so contributors can experiment with models and features.
- Encourage reproducible experiments with deterministic data splits and clear instructions.

What this repository contains
- Scripts to download historical OHLCV (open, high, low, close, volume) data for tickers.
- Utilities to compute commonly used features and indicators.
- Example training scripts for baseline models (classical and machine learning).
- A small Flask app to serve predictions and examples of how to call the model.
- Documentation and examples for evaluation and experiment tracking.

Quick start (local)
1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Fetch example data:
   ```
   python src/fetch_data.py --ticker AAPL --start 2015-01-01 --end 2024-12-31 --out data/AAPL.csv
   ```

3. Compute features:
   ```
   python src/features.py --input data/AAPL.csv --output data/AAPL_features.csv
   ```

4. Train a model:
   ```
   python src/train.py --data data/AAPL_features.csv --model_dir models/aapl_baseline
   ```

5. Serve predictions with Flask:
   ```
   FLASK_APP=app.py flask run
   # or
   python app.py
   ```

Best practices and notes
- Use time-aware train/validation/test splits (walk-forward or rolling windows) rather than random shuffles.
- Fit scalers and any preprocessing using only training data, then apply to validation/test sets.
- Track experiments: record parameter settings, data ranges, and random seeds to make results reproducible.
- Start with simple baselines before moving to more complex models.

Project structure
- app.py                 # Flask application
- src/
  - fetch_data.py        # data download utilities
  - features.py          # feature engineering
  - train.py             # training and evaluation scripts
  - predict.py           # inference helpers
- data/                  # raw and processed CSVs
- models/                # saved model artifacts
- requirements.txt
- README.md

Contributing
Contributions are welcome. Open an issue to discuss larger changes, or submit small, focused pull requests. Please include tests or clear usage notes for code additions.

License
MIT