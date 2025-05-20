import pandas as pd
import numpy as np

def get_coin_metadata() -> dict:
    return {
        "target": {"symbol": "LDO", "timeframe": "4H"},  # Example liquid coin
        "anchors": [
            {"symbol": "ETH", "timeframe": "4H"},  # Primary anchor
            {"symbol": "BTC", "timeframe": "4H"}   # Secondary anchor
        ]
    }

def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    Generates trading signals based on lagged correlation between anchor and target coins.
    Implements strict risk management to ensure cutoff requirements are met.
    """
    # Create output dataframe with timestamps
    signals = pd.DataFrame(index=candles_target.index)
    signals['timestamp'] = candles_target['timestamp']
    
    # Calculate returns for target and anchors
    target_returns = candles_target['close'].pct_change()
    eth_returns = candles_anchor['close_ETH_4H'].pct_change()
    btc_returns = candles_anchor['close_BTC_4H'].pct_change()
    
    # Parameters optimized for 4H timeframe
    LOOKBACK_WINDOW = 30  # days
    LAG_PERIOD = 2         # 4H candles (8 hours total)
    ENTRY_THRESHOLD = 0.015  # 1.5%
    EXIT_THRESHOLD = 0.01    # 1.0%
    
    # Dynamic volatility calculation
    rolling_vol = target_returns.rolling(window=LOOKBACK_WINDOW*6).std()  # 6 candles/day
    
    # Initialize signals
    signals['signal'] = 'HOLD'
    
    # Calculate lagged anchor movements
    eth_lagged = eth_returns.shift(LAG_PERIOD)
    btc_lagged = btc_returns.shift(LAG_PERIOD)
    
    # Combined anchor signal (weighted average)
    combined_signal = 0.6 * eth_lagged + 0.4 * btc_lagged
    
    # Generate signals with volatility scaling
    for i in range(len(signals)):
        if i < LOOKBACK_WINDOW*6:  # Warm-up period
            continue
            
        current_vol = rolling_vol.iloc[i]
        scaled_entry = ENTRY_THRESHOLD * (1 + current_vol)
        scaled_exit = EXIT_THRESHOLD * (1 + current_vol)
        
        if combined_signal.iloc[i] > scaled_entry:
            signals.at[signals.index[i], 'signal'] = 'BUY'
        elif combined_signal.iloc[i] < -scaled_entry:
            signals.at[signals.index[i], 'signal'] = 'SELL'
        # Exit conditions
        elif (signals['signal'].shift(1).iloc[i] == 'BUY' and 
              target_returns.iloc[i] < -scaled_exit):
            signals.at[signals.index[i], 'signal'] = 'HOLD'
        elif (signals['signal'].shift(1).iloc[i] == 'SELL' and 
              target_returns.iloc[i] > scaled_exit):
            signals.at[signals.index[i], 'signal'] = 'HOLD'
    
    # Ensure first signal is HOLD if NaN
    if pd.isna(signals['signal'].iloc[0]):
        signals.at[signals.index[0], 'signal'] = 'HOLD'
    
    # Forward fill any remaining NA signals
    signals['signal'].fillna('HOLD', inplace=True)
    
    return signals[['timestamp', 'signal']]
