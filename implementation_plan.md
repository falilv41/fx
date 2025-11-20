# Performance Visualization Implementation Plan

## Goal
Visualize the model's historical performance (accuracy, win rate, trade locations) on the web interface.

## Backend (`web/app.py`)
1.  **New Endpoint**: `/api/performance`
    *   **Input**: Timeframe (default: last 3 months).
    *   **Process**:
        *   Fetch historical data (e.g., 3 months of 1h data).
        *   Generate features.
        *   Run predictions with the loaded model.
        *   Apply the trading threshold (0.52).
        *   Compare predictions with actual next-hour outcomes.
        *   Calculate metrics: Win Rate, Total Return, Number of Trades.
        *   Identify individual trades: Entry Time, Type (Long/Short), Entry Price, Result (Win/Loss).
    *   **Output**: JSON containing metrics, trade list, and OHLC data for the chart.

## Frontend (`web/templates/index.html`)
1.  **UI Updates**:
    *   Add a "Performance Analysis" section below the main prediction area.
    *   "Analyze Past 3 Months" button.
    *   **Metrics Display**: Cards for Win Rate, Total Return, Trade Count.
    *   **Interactive Chart**:
        *   Candlestick chart for the backtest period.
        *   **Markers**:
            *   Green Triangle Up: Winning Long
            *   Red Triangle Up: Losing Long
            *   Green Triangle Down: Winning Short
            *   Red Triangle Down: Losing Short
            *   (Maybe simplified: Green Circle = Win, Red X = Loss, with hover text for direction)

## Steps
1.  Update `web/app.py` to implement the backtest logic and API.
2.  Update `web/templates/index.html` to consume the API and render the advanced chart.
