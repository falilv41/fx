import pandas as pd
import numpy as np
import lightgbm as lgb
from src.features import create_features
from src.backtest import calculate_trading_profit
import matplotlib.pyplot as plt
import joblib

def run_best_model():
    print("Loading data...")
    df = pd.read_csv('data/USDJPY_1h.csv')
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
    print("Generating features...")
    df = create_features(df)
    
    # Create target: 1 if Up, 0 if Down
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaNs
    df = df.dropna()
    
    # Features to use
    exclude_cols = ['Datetime', 'target', 'Close', 'High', 'Low', 'Open', 'Volume', 'Price', 'Price_integer', 'Price_rounded_01', 'Price_rounded_001']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter out raw price columns
    final_features = []
    for col in feature_cols:
        is_raw = False
        if any(k in col for k in ['SMA_', 'EMA_', 'BB_Upper', 'BB_Lower']) and not any(k in col for k in ['ratio', 'diff', 'width', 'position']):
             is_raw = True
        if 'lag' in col and 'Return' not in col:
             is_raw = True
        
        if not is_raw:
            final_features.append(col)
            
    print(f"Using {len(final_features)} features.")
    
    # Split data (Last 20% for Test)
    n = len(df)
    test_len = int(n * 0.2)
    train_len = n - test_len
    
    train_df = df.iloc[:train_len]
    test_df = df.iloc[train_len:]
    
    X_train = train_df[final_features]
    y_train = train_df['target']
    X_test = test_df[final_features]
    y_test = test_df['target']
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Train Model
    print("Training LightGBM model...")
    
    # Use a validation set for early stopping
    valid_split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:valid_split], X_train.iloc[valid_split:]
    y_tr, y_val = y_train.iloc[:valid_split], y_train.iloc[valid_split:]
    
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predict
    preds_prob = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Evaluate
    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = roc_auc_score(y_test, preds_prob)
    acc = accuracy_score(y_test, (preds_prob > 0.5).astype(int))
    print(f"Test AUC: {auc:.6f}, Accuracy: {acc:.6f}")
    
    # Backtest with best threshold
    threshold = 0.52
    print(f"Running backtest with threshold {threshold}...")
    
    current_prices = test_df['Close'].values
    actual_next_price = test_df['Close'].shift(-1).values
    if np.isnan(actual_next_price[-1]):
         actual_next_price[-1] = current_prices[-1]
         
    preds_price = current_prices.copy()
    for i in range(len(preds_prob)):
        if preds_prob[i] > threshold:
            preds_price[i] = current_prices[i] + 1.0 # Long
        elif preds_prob[i] < 1 - threshold:
            preds_price[i] = current_prices[i] - 1.0 # Short
        else:
            preds_price[i] = current_prices[i] # Neutral
            
    results = calculate_trading_profit(test_df, actual_next_price, preds_price)
    
    print("="*30)
    print(f"Final Capital: {results['final_capital']:,.0f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Trades: {results['total_trades']}")
    print("="*30)
    
    # Save model
    joblib.dump(model, 'model_lgb.pkl')
    print("Model saved to model_lgb.pkl")

if __name__ == "__main__":
    run_best_model()
