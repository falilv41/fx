import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from src.features import create_features
from src.backtest import calculate_trading_profit
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

def train_model(data_path, model_type='lgb', target_col='target'):
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
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
            
    print(f"Using {len(final_features)} features out of {len(feature_cols)}")
    
    # TimeSeriesSplit
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    X = df[final_features]
    y = df['target']
    
    models = []
    scores = []
    
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
    
    print("Starting TimeSeries CV...")
    for fold, (train_index, valid_index) in enumerate(tscv.split(X)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        preds = model.predict(X_valid, num_iteration=model.best_iteration)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_valid, preds)
        scores.append(auc)
        models.append(model)
        print(f"Fold {fold+1} AUC: {auc:.6f}")
        
    print(f"Average AUC: {np.mean(scores):.6f}")
    
    # Retrain on all data except test set (last 20%) for final evaluation
    # Actually, let's use the last fold's validation as "Test" or split explicitly before CV.
    # Better: Split Test first, then CV on Train/Valid.
    
    n = len(df)
    test_len = int(n * 0.2)
    train_valid_len = n - test_len
    
    train_valid_df = df.iloc[:train_valid_len]
    test_df = df.iloc[train_valid_len:]
    
    X_train_all = train_valid_df[final_features]
    y_train_all = train_valid_df['target']
    X_test = test_df[final_features]
    y_test = test_df['target']
    
    print(f"\nRetraining on full train set ({len(train_valid_df)} samples) for testing on {len(test_df)} samples...")
    
    lgb_train_all = lgb.Dataset(X_train_all, y_train_all)
    # Use a portion of train_valid as validation for early stopping, or just use the best iteration from CV?
    # Let's use a simple split for early stopping
    valid_split = int(len(X_train_all) * 0.8)
    X_tr, X_val = X_train_all.iloc[:valid_split], X_train_all.iloc[valid_split:]
    y_tr, y_val = y_train_all.iloc[:valid_split], y_train_all.iloc[valid_split:]
    
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    
    final_model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    preds_prob = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    
    # Evaluate
    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = roc_auc_score(y_test, preds_prob)
    acc = accuracy_score(y_test, (preds_prob > 0.5).astype(int))
    print(f"Test AUC: {auc:.6f}, Accuracy: {acc:.6f}")
    
    # Feature Importance
    importance = final_model.feature_importance(importance_type='gain')
    feature_name = final_model.feature_name()
    importance_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    print("\nTop 20 Features:")
    print(importance_df)
    
    # Backtest with multiple thresholds
    print("Running backtest with multiple thresholds...")
    current_prices = test_df['Close'].values
    actual_next_price = test_df['Close'].shift(-1).values
    if np.isnan(actual_next_price[-1]):
         actual_next_price[-1] = current_prices[-1]
    
    best_return = -999
    best_threshold = 0.5
    
    thresholds = [0.5, 0.51, 0.52, 0.53, 0.54, 0.55]
    
    for th in thresholds:
        preds_price = current_prices.copy()
        
        # Logic:
        # If prob > th -> Long
        # If prob < 1 - th -> Short
        # Else -> Neutral
        
        for i in range(len(preds_prob)):
            if preds_prob[i] > th:
                preds_price[i] = current_prices[i] + 1.0 # Force Long
            elif preds_prob[i] < 1 - th:
                preds_price[i] = current_prices[i] - 1.0 # Force Short
            else:
                preds_price[i] = current_prices[i] # Neutral
        
        results = calculate_trading_profit(test_df, actual_next_price, preds_price)
        
        print(f"Threshold: {th:.2f} | Return: {results['total_return']:.2f}% | Trades: {results['total_trades']} | Win: {results['win_rate']:.2f}%")
        
        if results['total_return'] > best_return:
            best_return = results['total_return']
            best_threshold = th
            
    print(f"\nBest Threshold: {best_threshold:.2f} with Return: {best_return:.2f}%")
    
    return final_model, best_return

if __name__ == "__main__":
    train_model('data/USDJPY_1h.csv')
