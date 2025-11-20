import sys
import os
# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import pytz
from src.features import create_features

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_lgb.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def get_recent_data(period='5d', interval='1h'):
    """Fetch recent data from Yahoo Finance"""
    ticker = "USDJPY=X"
    # Fetch slightly more data to ensure we have enough for lag features
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    
    # Handle MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    # Convert timezone to JST
    jst = pytz.timezone('Asia/Tokyo')
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    data = data.tz_convert(jst)
    
    data = data.reset_index()
    data.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True)
    
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/market_data')
def market_data():
    try:
        # Get last 48 hours for display + buffer for calculation
        df = get_recent_data(period='5d', interval='1h')
        
        # Filter for display (last 24h)
        last_24h = df.tail(24).copy()
        
        # Format for JSON
        result = []
        for _, row in last_24h.iterrows():
            result.append({
                'time': row['Datetime'].strftime('%Y-%m-%d %H:%M'),
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close']
            })
            
        return jsonify({'status': 'success', 'data': result, 'current_price': df['Close'].iloc[-1]})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/predict')
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'})
        
    try:
        # Get data sufficient for feature engineering (need at least 200 rows for SMA_200)
        df = get_recent_data(period='1mo', interval='1h')
        
        # Feature Engineering
        df_features = create_features(df)
        
        # Prepare features for the latest candle
        # We want to predict the NEXT movement based on the LATEST available closed candle.
        # However, yfinance '1h' data usually returns the current incomplete candle as the last row if market is open.
        # We should check the time. If the last row is the current hour, we use it to predict the next hour?
        # Or do we wait for close?
        # For this demo, we'll take the very last row available as the "current state".
        
        latest_data = df_features.iloc[[-1]].copy()
        
        # Features list (must match training)
        # We need to filter columns exactly as in training
        # Load feature names from model if possible, or replicate logic
        # LightGBM model object stores feature_name
        model_features = model.feature_name()
        
        # Check if all features exist
        missing_features = [f for f in model_features if f not in latest_data.columns]
        if missing_features:
            return jsonify({'status': 'error', 'message': f'Missing features: {missing_features}'})
            
        X_pred = latest_data[model_features]
        
        # Predict Probability
        prob = model.predict(X_pred)[0]
        
        # Threshold logic (from our optimization)
        threshold = 0.52
        direction = "Neutral"
        confidence = 0.0
        
        if prob > threshold:
            direction = "UP"
            confidence = prob
        elif prob < 1 - threshold:
            direction = "DOWN"
            confidence = 1 - prob
        else:
            direction = "WAIT"
            confidence = max(prob, 1-prob) # Just show the higher prob
            
        return jsonify({
            'status': 'success',
            'prediction': direction,
            'probability': float(prob),
            'confidence_display': f"{confidence*100:.1f}%",
            'timestamp': latest_data['Datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/performance')
def performance():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'})
        
    try:
        # Fetch 3 months of data
        df = get_recent_data(period='3mo', interval='1h')
        
        # Feature Engineering
        df_features = create_features(df)
        
        # Create target for evaluation (Next Close > Current Close)
        df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
        df_features['next_close'] = df_features['Close'].shift(-1)
        
        # Drop NaNs created by features and target
        df_eval = df_features.dropna().copy()
        
        # Prepare features
        model_features = model.feature_name()
        missing_features = [f for f in model_features if f not in df_eval.columns]
        if missing_features:
            return jsonify({'status': 'error', 'message': f'Missing features: {missing_features}'})
            
        X_eval = df_eval[model_features]
        
        # Predict
        probs = model.predict(X_eval)
        
        # Backtest Logic
        threshold = 0.52
        trades = []
        capital = 1000000
        initial_capital = capital
        wins = 0
        total_trades = 0
        
        # For chart, we need the full OHLC data of the eval period
        chart_data = []
        for _, row in df_eval.iterrows():
            chart_data.append({
                'time': row['Datetime'].strftime('%Y-%m-%d %H:%M'),
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close']
            })
            
        # Iterate through predictions
        for i in range(len(df_eval)):
            prob = probs[i]
            row = df_eval.iloc[i]
            current_price = row['Close']
            next_price = row['next_close']
            timestamp = row['Datetime'].strftime('%Y-%m-%d %H:%M')
            
            action = None
            if prob > threshold:
                action = 'LONG'
            elif prob < 1 - threshold:
                action = 'SHORT'
                
            if action:
                total_trades += 1
                profit = 0
                result = 'LOSS'
                
                # Simple 1h hold strategy
                if action == 'LONG':
                    profit = next_price - current_price - 0.002 # Spread
                else:
                    profit = current_price - next_price - 0.002 # Spread
                    
                if profit > 0:
                    wins += 1
                    result = 'WIN'
                
                # Position sizing (fixed lots or capital %? Let's say 1 lot = 10,000 units)
                # Profit in JPY per unit * units
                units = 10000
                trade_profit = profit * units
                capital += trade_profit
                
                trades.append({
                    'time': timestamp,
                    'action': action,
                    'price': current_price,
                    'prob': float(prob),
                    'result': result,
                    'profit': trade_profit
                })
                
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_return = ((capital - initial_capital) / initial_capital) * 100
        
        return jsonify({
            'status': 'success',
            'metrics': {
                'win_rate': f"{win_rate:.2f}%",
                'total_return': f"{total_return:.2f}%",
                'total_trades': total_trades,
                'final_capital': f"¥{capital:,.0f}"
            },
            'trades': trades,
            'chart_data': chart_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)

