import pandas as pd
import numpy as np

def calculate_trading_profit(test_df, y_test, y_test_pred, spread=0.002, initial_capital=1000000):
    """
    Calculate trading profit based on predictions.
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        Test data containing 'Datetime' and 'Close'
    y_test : pandas.Series or numpy.ndarray
        Actual next prices (or target values)
    y_test_pred : numpy.ndarray
        Predicted next prices (or target values)
    spread : float
        Spread cost (default 0.002 JPY)
    initial_capital : float
        Initial capital (default 1,000,000 JPY)
        
    Returns:
    --------
    dict : Trading statistics
    """
    # Ensure inputs are numpy arrays
    current_prices = test_df['Close'].values
    
    # If y_test is Series, convert to values
    if isinstance(y_test, pd.Series):
        actual_next_prices = y_test.values
    else:
        actual_next_prices = y_test
        
    predicted_next_prices = y_test_pred
    
    capital = initial_capital
    position = None  # None, 'long', 'short'
    entry_price = None
    trades = []
    
    winning_trades = 0
    losing_trades = 0
    total_trades = 0
    total_spread_cost = 0
    
    # Iterate through time steps
    for i in range(len(current_prices) - 1):
        current_price = current_prices[i]
        predicted_next = predicted_next_prices[i]
        actual_next = actual_next_prices[i]
        
        # Signal generation
        # If predicting price:
        signal = None
        if predicted_next > current_price + spread: # Add a threshold?
            signal = 'long'
        elif predicted_next < current_price - spread:
            signal = 'short'
        
        # Close existing position
        if position is not None:
            # Calculate profit
            if position == 'long':
                # Sell at actual_next - spread/2
                exit_price = actual_next - spread/2
                # Entry was entry_price + spread/2
                entry_cost = entry_price + spread/2
                
                profit = (exit_price - entry_cost) * (capital / entry_cost) # Simplified compounding
                # More accurate: Profit = (Exit - Entry) * Units
                # Units = Capital / Entry
                units = capital / entry_cost
                profit_amount = (exit_price - entry_cost) * units
                
                capital += profit_amount
                
                trades.append({
                    'type': 'long',
                    'entry': entry_cost,
                    'exit': exit_price,
                    'profit': profit_amount
                })
                
            elif position == 'short':
                # Buy back at actual_next + spread/2
                exit_price = actual_next + spread/2
                # Entry was entry_price - spread/2
                entry_cost = entry_price - spread/2
                
                # Profit for short: (Entry - Exit) * Units
                units = capital / entry_cost
                profit_amount = (entry_cost - exit_price) * units
                
                capital += profit_amount
                
                trades.append({
                    'type': 'short',
                    'entry': entry_cost,
                    'exit': exit_price,
                    'profit': profit_amount
                })
            
            if trades[-1]['profit'] > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            total_trades += 1
            
            position = None
            
        # Open new position
        if signal is not None:
            position = signal
            entry_price = current_price
            
    # Close final position
    if position is not None:
        actual_final = actual_next_prices[-1]
        if position == 'long':
            exit_price = actual_final - spread/2
            entry_cost = entry_price + spread/2
            units = capital / entry_cost
            profit_amount = (exit_price - entry_cost) * units
            capital += profit_amount
        elif position == 'short':
            exit_price = actual_final + spread/2
            entry_cost = entry_price - spread/2
            units = capital / entry_cost
            profit_amount = (entry_cost - exit_price) * units
            capital += profit_amount
            
    total_return = (capital - initial_capital) / initial_capital * 100
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'trades': trades
    }
