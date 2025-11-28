import numpy as np
import pandas as pd

def remove_outliers(ts, debug = False):
    # min-max scaling
    ts_min, ts_max = np.min(ts), np.max(ts)
    ts_scaled = (ts - ts_min) / (ts_max - ts_min)
    
    ts_diffs = ts_scaled.diff().reindex(ts_scaled.index)
    start_idx = (ts_scaled.notna()) & (ts_diffs.isna() & (ts_scaled.index != ts_scaled.index[-1]))

    # --- Manually assign diff to the first value after a gap ---
    # Extract the actual date values where the condition is True
    dates = ts_scaled.index[start_idx]  # these are your DatetimeIndex entries
    # Shift those dates by one month
    shifted_dates = dates + pd.DateOffset(months=1)
    # Assign the shifted values back; the .values ensures correct one-to-one assignment
    ts_diffs.loc[dates] = ts_diffs.loc[shifted_dates].values   

    Q1 = ts_diffs.quantile(0.25)
    Q3 = ts_diffs.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (ts_diffs < (Q1 - 2.5 * IQR)) | (ts_diffs > (Q3 + 2.5 * IQR))
    
    # Ensure the new value is also outside a reasonable local range
    window_size = 12  # Adjust based on expected local patterns
    
    # Compute rolling min/max for past values (shift forward)
    rolling_min_past = ts_scaled.rolling(window=window_size, min_periods=1, closed='left').min().shift(1)
    rolling_max_past = ts_scaled.rolling(window=window_size, min_periods=1, closed='left').max().shift(1)
    # Compute rolling min/max for future values (shift backward)
    rolling_min_future = ts_scaled[::-1].rolling(window=window_size, min_periods=1, closed='left').min().shift(1)[::-1]
    rolling_max_future = ts_scaled[::-1].rolling(window=window_size, min_periods=1, closed='left').max().shift(1)[::-1]

    rolling_min = pd.concat([rolling_min_past, rolling_min_future], axis=1).min(axis=1, skipna=True)
    rolling_max = pd.concat([rolling_max_past, rolling_max_future], axis=1).max(axis=1, skipna=True)

    contextual_outliers = outliers & ~((ts_scaled > rolling_min - 0.4) & (ts_scaled < rolling_max + 0.4))

    ts.loc[contextual_outliers] = np.nan

    #Results
    if debug:
        result_df = pd.DataFrame({
            'Value (scaled)': ts_scaled,
            'Diff': ts_diffs,
            'Rolling min': rolling_min,
            'Rolling max': rolling_max,
            'Low threshold': rolling_min - 0.4,
            'High threshold': rolling_max + 0.4,
            'Global outlier': outliers,
            'Contextual outlier': contextual_outliers
        })
        
        result_df.to_csv('outlier_test.csv')
    return ts, contextual_outliers.sum() > 0
