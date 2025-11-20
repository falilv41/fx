# FX Prediction Model Walkthrough

## Goal
Build a high-accuracy USD/JPY prediction model and optimize it for trading profit.

## Approach
1.  **Data**: Used 1-hour USD/JPY data (approx 2 years).
2.  **Features**:
    *   Moving Averages (SMA, EMA)
    *   Oscillators (RSI, Stochastic, Williams %R)
    *   Volatility (ATR, Bollinger Bands)
    *   **Multi-timeframe Returns** (1h, 4h, 24h, 48h) - *Key for trend capture*
    *   Psychological Levels (Distance from 100, 150, etc.)
3.  **Model**: LightGBM Classifier (Binary: Up/Down).
4.  **Validation**: TimeSeriesSplit to prevent lookahead bias.
5.  **Strategy**:
    *   Predict probability of "Up" movement.
    *   Trade only when probability > **0.52** (High confidence).
    *   Hold for 1 hour.

## Results (Test Set)
*   **Period**: Last 20% of data (approx 5 months).
*   **Total Return**: **+4.16%**
*   **Win Rate**: **56.91%**
*   **Total Trades**: 304 (approx 2 trades/day)
*   **Final Capital**: 1,041,584 JPY (from 1,000,000 JPY)

## Key Findings
*   **Raw Price Features**: Caused non-stationarity issues. Removed them in favor of returns, diffs, and ratios.
*   **Regression vs Classification**: Classification (Up/Down) worked better than Regression (Return amount) because the noise in return magnitude is high.
*   **Threshold Optimization**: Filtering trades with a probability threshold of **0.52** significantly improved the win rate and risk-adjusted return compared to trading every signal.
*   **Top Features**:
    1.  `Return_48h`: 48-hour momentum is a strong predictor.
    2.  `Volatility_30`: Market volatility regime matters.
    3.  `RSI_7`: Short-term overbought/oversold conditions.

## Files
*   `src/features.py`: Feature engineering logic.
*   `src/train.py`: Training and experimentation script.
*   `src/run_best_model.py`: Final script to run the best configuration.
*   `model_lgb.pkl`: Saved model artifact.
