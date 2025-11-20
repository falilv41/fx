import pandas as pd
import numpy as np

def create_features(df):
    """
    Create technical indicators and other features for FX data.
    """
    df = df.copy()
    
    # Ensure Datetime is datetime object
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Hour'] = df['Datetime'].dt.hour
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        df['DayOfMonth'] = df['Datetime'].dt.day
        df['Month'] = df['Datetime'].dt.month

    # === 1. Moving Averages (SMA) ===
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Close_SMA_{window}_ratio'] = df['Close'] / df[f'SMA_{window}']
        df[f'Close_SMA_{window}_diff'] = df['Close'] - df[f'SMA_{window}']
        df[f'Close_SMA_{window}_diff_pct'] = (df['Close'] - df[f'SMA_{window}']) / df[f'SMA_{window}'] * 100
    
    # === 2. Exponential Moving Averages (EMA) ===
    for span in [5, 10, 20, 50]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        df[f'Close_EMA_{span}_ratio'] = df['Close'] / df[f'EMA_{span}']
        df[f'Close_EMA_{span}_diff'] = df['Close'] - df[f'EMA_{span}']
    
    # === 3. RSI ===
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    for window in [7, 14, 21]:
        df[f'RSI_{window}'] = calculate_rsi(df['Close'], window=window)
    
    # === 5. Volatility ===
    for window in [5, 10, 20, 30]:
        df[f'Volatility_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Volatility_{window}_pct'] = df[f'Volatility_{window}'] / df['Close'] * 100
    
    # ATR
    def calculate_atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    for window in [14, 20]:
        df[f'ATR_{window}'] = calculate_atr(df['High'], df['Low'], df['Close'], window)
    
    # === 6. Price Action ===
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Range_Pct'] = df['Price_Range'] / df['Close'] * 100
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Body_Size_Pct'] = df['Body_Size'] / df['Close'] * 100
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Change_Pct'] = df['Price_Change'] / df['Open'] * 100
    
    # === 7. MACD ===
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # === 8. Bollinger Bands ===
    for window in [20, 50]:
        ma = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        df[f'BB_Upper_{window}'] = ma + (2 * std)
        df[f'BB_Lower_{window}'] = ma - (2 * std)
        df[f'BB_Width_{window}'] = df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}']
        df[f'BB_Position_{window}'] = (df['Close'] - df[f'BB_Lower_{window}']) / (df[f'BB_Width_{window}'] + 1e-10)
    
    # === 9. Lag Features ===
    # Added more lags
    for lag in [1, 2, 3, 5, 10, 24]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'High_lag_{lag}'] = df['High'].shift(lag)
        df[f'Low_lag_{lag}'] = df['Low'].shift(lag)
        # Return lags
        df[f'Return_lag_{lag}'] = df['Close'].pct_change(lag)
        
    # Multi-timeframe returns
    df['Return_4h'] = df['Close'].pct_change(4)
    df['Return_24h'] = df['Close'].pct_change(24)
    df['Return_48h'] = df['Close'].pct_change(48)
    
    # === 11. Other Indicators ===
    # Stochastic
    def calculate_stochastic(high, low, close, window=14):
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        return k_percent
    
    df['Stoch_K'] = calculate_stochastic(df['High'], df['Low'], df['Close'], window=14)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Williams %R
    def calculate_williams_r(high, low, close, window=14):
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    df['Williams_R'] = calculate_williams_r(df['High'], df['Low'], df['Close'], window=14)
    
    # === 14. Psychological Levels ===
    df['Price_integer'] = df['Close'].apply(lambda x: int(x))
    psychological_levels = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    
    for level in psychological_levels:
        df[f'Distance_from_level_{level}'] = df['Close'] - level
    
    # Drop NaN values created by rolling windows
    # df = df.dropna() # Don't drop here, let the trainer handle it or drop at the end
    
    return df
