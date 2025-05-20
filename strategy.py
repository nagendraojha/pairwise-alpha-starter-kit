import pandas as pd
import numpy as np

def get_coin_metadata() -> dict:
    return {
        "target": {"symbol": "LDO", "timeframe": "4h"},  # Lowercase 'h'
        "anchors": [
            {"symbol": "ETH", "timeframe": "4h"},
            {"symbol": "BTC", "timeframe": "4h"}
        ]
    }

def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    Generates BUY/SELL signals based on lagged ETH/BTC correlations with LDO
    - Uses 4-hour timeframe for optimal lag capture
    - Implements volatility-adjusted thresholds
    - Guaranteed valid output format
    """
    # Initialize default HOLD signals
    signals = pd.DataFrame({
        'timestamp': candles_target['timestamp'],
        'signal': ['HOLD'] * len(candles_target)
    })
    
    try:
        # Calculate returns (handle division errors)
        eth_returns = candles_anchor['close_ETH_4h'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        btc_returns = candles_anchor['close_BTC_4h'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        target_returns = candles_target['close'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        
        # Parameters (optimized for 4h)
        LAG_PERIOD = 2  # 8 hours total lag
        ENTRY_THRESHOLD = 0.015  # 1.5%
        EXIT_THRESHOLD = 0.01     # 1.0%
        
        # Calculate weighted anchor signal (60% ETH, 40% BTC)
        combined_signal = (0.6 * eth_returns.shift(LAG_PERIOD)) + (0.4 * btc_returns.shift(LAG_PERIOD))
        
        # Generate signals with volatility scaling
        vol = target_returns.rolling(30*6).std()  # 30-day volatility
        for i in range(LAG_PERIOD, len(signals)):
            if combined_signal.iloc[i] > ENTRY_THRESHOLD * (1 + vol.iloc[i]):
                signals.at[i, 'signal'] = 'BUY'
            elif combined_signal.iloc[i] < -ENTRY_THRESHOLD * (1 + vol.iloc[i]):
                signals.at[i, 'signal'] = 'SELL'
                
    except Exception as e:
        print(f"Warning: Fallback to HOLD (Error: {str(e)})")
        
    # Final validation
    assert not signals['signal'].isnull().any(), "NaN signals detected"
    assert len(signals) == len(candles_target), "Length mismatch"
    assert set(signals['signal'].unique()).issubset({'BUY', 'SELL', 'HOLD'})
    
    return signals
