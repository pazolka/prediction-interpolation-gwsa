import numpy as np
import pandas as pd
import statsmodels.api as sm

import ruptures as rpt
from statsmodels.tsa.seasonal import STL

### MAIN FUNCTION ###
def detect_well_change(ts, pen = 14, debug = False):
    # min-max scaling
    ts_min, ts_max = np.min(ts), np.max(ts)
    ts_scaled = (ts - ts_min)/(ts_max - ts_min)

    # simple interpolation to prevent the method from breaking
    ts_filled = ts_scaled.interpolate(method='polynomial', order=3).dropna()
    
    # change point detection on filled ts
    break_indices = detect_change_points(ts_filled, pen=pen)
    # filter breakpoints (filled ts)
    filtered_breaks = []
    window = 12

    for pos, b in enumerate(break_indices):
        b_date = ts_filled.index[b] # break index as date

        # required offsets
        l12_start_date = b_date - pd.DateOffset(months=window)
        r12_end_date = b_date + pd.DateOffset(months=window)
        r48_end_date = b_date + pd.DateOffset(months=3*window)

        l48_start_date = b_date - pd.DateOffset(months=4*window+3)
        l3_end_date = b_date - pd.DateOffset(months=3)

        ext_start_date = ts_filled.index[0] if pos == 0 else ts_filled.index[break_indices[pos-1]]
        ext_end_date = ts_filled.index[-1] if pos == len(break_indices)-1 else ts_filled.index[break_indices[pos+1]]

        ##################################################################################################################
        #
        # basic checks:
        #     1) mean diff between left 12m and right 12m windows, relative to left mean shouldn't exceed 0.7
        #       - if any of the means is nan -> large gap
        #            - recalculate on the nearest 12m left and right windows
        #     2) quantile difference between left and right Q1 (Q3), relative to left (right) IQR shound't exceed 1.5 
        #     3) IQR shoulnd't differ by more than the factor of 2 (both sided)
        #     4) min (max) values shouldn't differ by more than the factor of 3 (both sided)
        #
        ##################################################################################################################

        # windows for means and quantiles
        window_l = ts_scaled[l12_start_date:b_date] # breakpoint - 12 : breakpoint
        window_r = ts_scaled[b_date:r12_end_date] # breakpoint : breakpoint + 12
        
        # calculate means on each side of the break point
        mean_left = window_l.mean() if len(window_l.dropna()) > 3 else np.nan
        mean_right = window_r.mean() if len(window_r.dropna()) > 3 else np.nan

        large_gap_detected = np.isnan(mean_left) or np.isnan(mean_right)

        # if a large gap is detected,
        # find the closest window of 12 months with max of 6 nans on both sides of the breakpoint
        # find a mean and quantiles based on that window
        # use those windows to calculate slopes
        if large_gap_detected:
            l12_start_date, window_l = find_12_window_left(ts_scaled, b_date)
            r12_end_date, window_r = find_12_window_right(ts_scaled, b_date)

            if pd.isna(l12_start_date):
                # there is another breakpoint on the left very close to the current break - handle this case manually
                print(f'handle manually: {ts_scaled.name} | {b_date}')
                continue 
            if pd.isna(r12_end_date):
                # likely reached the end of a timeseries - create change point to get rid of an outlier
                filtered_breaks.append(ts_scaled.index.get_loc(b_date))
                continue

            l12_end_date = l12_start_date + pd.DateOffset(months=12)
            r12_start_date = r12_end_date - pd.DateOffset(months=12)
            r48_end_date = r12_end_date + pd.DateOffset(months=24)

            # check if another breakpoint is very close to the left
            temp = l12_start_date - pd.DateOffset(months=3*window)
            l48_start_date = ts_filled.index[break_indices[pos-1]] \
                if pos > 0 and ts_filled.index[break_indices[pos-1]] > temp \
                else temp

            # in this edge case, these are equal bc no need to shorten if there is a gap / no jump
            l3_end_date = b_date

            mean_left = window_l.mean()
            mean_right = window_r.mean()

            left_unfilled = ts_scaled[l48_start_date:l12_end_date]
            right_unfilled = ts_scaled[r12_start_date:r48_end_date]

            # edge cases where break points are very close to each other
            # the same left (right) interval for both break points
            if (~pd.isna(left_unfilled)).sum() < 3:
                l48_start_date = temp
                left_unfilled = ts_scaled[l48_start_date:l12_end_date]
            
            left_filled = left_unfilled.interpolate(method='polynomial', order=3).dropna()
            right_filled = right_unfilled.interpolate(method='polynomial', order=3).dropna()

        else:
            # divide ts around the break point and fill each part individually
            left_filled = ts_scaled[ext_start_date:b_date].interpolate(method='polynomial', order=3)
            right_filled = ts_scaled[b_date:ext_end_date].interpolate(method='polynomial', order=3)

        # calculate Q1 and Q3 and min/max on each side of the break point
        min_l = window_l.min()
        max_l = window_l.max()
        Q1_l = window_l.quantile(0.25)
        Q3_l = window_l.quantile(0.75)

        min_r = window_r.min()
        max_r = window_r.max()
        Q1_r = window_r.quantile(0.25)
        Q3_r = window_r.quantile(0.75)

        if ( Q1_l == Q3_l ) or ( Q1_r == Q3_r ):
            # a rare event when there was a human error / same value is entered multiple times
            filtered_breaks.append(ts_scaled.index.get_loc(b_date))
            continue


        large_mean_difference = abs((mean_left - mean_right) / mean_left) > 0.5 # relative to mean_left
        sudden_quantile_shift = abs(Q1_l-Q1_r)/abs(Q3_l-Q1_l) > 2 or abs(Q3_l-Q3_r)/abs(Q3_l-Q1_l) > 2
        
        sudden_iqr_change = (abs(Q3_l - Q1_l) / abs(Q3_r - Q1_r) > 1.9) or (abs(Q3_r - Q1_r) / abs(Q3_l - Q1_l) > 1.9)

        sudden_minmax_shift = (( abs(min_l-min_r)/abs(Q3_l-Q1_l) > 2 or abs(max_l-max_r)/abs(Q3_l-Q1_l) > 2 ) \
            or ( abs(min_r-min_l)/abs(Q3_r-Q1_r) > 2 or abs(max_r-max_l)/abs(Q3_r-Q1_r) > 2 )) or (min_r-min_l)/abs(Q3_r-Q1_r) > 1.5 # if there's any sudden 'rising'

        ######################################################
        #
        # prepare windows and slopes for advanced checks
        # - short left (48m - 3m)
        # - combined ( left 12m + right 12m )
        # - extended combined ( left 12m + right 48m )
        # - extended right ( LT till end / next break point )
        #
        ######################################################

        # shorten left windows slightly to exclude sudden jumps for long term trend assessment
        window_l_short = left_filled[l48_start_date:l3_end_date].dropna()
        # compute slope on left side of the break point
        slope_l_short, se_l_short = estimate_slope(window_l_short)

        window_combined_filled_full = pd.concat([left_filled, right_filled]).drop_duplicates() \
            .interpolate(method='polynomial', order=3).dropna()
        # compute slope for the combined 24-month window
        window_combined_filled = window_combined_filled_full[l12_start_date:r12_end_date]
        slope_combined, se_combined = estimate_slope(window_combined_filled)

        # compute slope for the extended 24 month right window
        window_combined_filled_ext = window_combined_filled_full[l12_start_date:r48_end_date]
        slope_combined_ext, se_combined_ext = estimate_slope(window_combined_filled_ext)

        # compute slope for the future
        right_ext_end = ts_filled.index[break_indices[pos+1] - 12] if pos < len(break_indices)-1 else None

        # if there is a large gap and another change point nearby 'overshadows' this operation - rely on the next change point 
        if (right_ext_end and right_filled.index[0] > right_ext_end ) :
            continue
        slope_long_term, se_lt = estimate_slope(right_filled[:right_ext_end].dropna())


        ##################################################################################################################
        #
        # advanced checks:
        #     5) is the pattern still disrupted if individually filled left and right windows are stitched together? 
        #     6) is the trend around the change point preserved? allowing for long term gradual decline
        #
        ##################################################################################################################

        disrupted_pattern = len(detect_change_points(pd.concat([left_filled[(b_date - pd.DateOffset(years=10)):], right_filled[:(b_date + pd.DateOffset(years=10))]]).dropna().drop_duplicates())) > 0

        long_term_gradual_decline = ( slope_long_term < -0.0001 and slope_combined_ext < 0  \
            and ( abs((slope_combined_ext - slope_long_term) / slope_combined_ext) < 0.8 \
                or abs((slope_combined_ext - slope_long_term) / slope_long_term) < 0.8)) \
            or ( slope_long_term < 0 and slope_combined_ext < 0  \
            and abs((slope_combined_ext - slope_long_term) / slope_combined_ext) < 0.3 ) \
            or ( (max(window_l.iloc[-3:]) > window_r.dropna()).mean() > 0.75 and abs((slope_combined_ext - slope_long_term) / slope_combined_ext) < 0.9 )

        trend_preserved = ( ( slope_l_short / slope_combined < 8 and slope_combined / slope_l_short < 4 and slope_combined / slope_l_short > 0 ) \
            or ( slope_l_short / slope_combined > -0.5 and slope_combined / slope_l_short > -0.5 and slope_combined / slope_l_short < 0 ) ) \
            or ( long_term_gradual_decline and slope_combined < 0 and slope_combined / slope_l_short > -8 )

        if debug == True:
            print(f'================ pos: {pos}, b : {b} | b_date: {b_date.date()}')
            print(f'LEFT mean: {mean_left:.4f}')
            print(f'RIGHT mean: {mean_right:.4f}')
            print(round(abs((mean_left - mean_right) / mean_left), 2))
            print(f'Sudden change in mean?: {abs((mean_left - mean_right) / mean_left) > 0.5}')
            print(f'LEFT Q1: {Q1_l:.4f}')
            print(f'LEFT Q3: {Q3_l:.4f}')
            print(f'RIGHT Q1: {Q1_r:.4f}')
            print(f'RIGHT Q3: {Q3_r:.4f}')
            print(f'LEFT IQR (12m): {(Q3_l-Q1_l):.4f}')
            print(f'RIGHT IQR (12m): {(Q3_r-Q1_r):.4f}')
            print(f'Q1 diff {abs(Q1_l-Q1_r):.4f}')
            print(f'Q3 diff {abs(Q3_l-Q3_r):.4f}')       
            print('-------')
            print(f'IQR_l larger than 2x IQR_r? {((Q3_l - Q1_l) / (Q3_r - Q1_r)):.4f}')
            print(f'left {(Q3_l - Q1_l):.4f} relative to right {(Q3_r - Q1_r):.4f}')
            print(f'IQR_r larger than 2x IQR_l? {((Q3_r - Q1_r) / (Q3_l - Q1_l)):.4f}')
            print(f'right {(Q3_r - Q1_r):.4f} relative to left {(Q3_l - Q1_l):.4f}')
            print('-------')
            print(f'min diff: {(min_r-min_l)/abs(Q3_r-Q1_r) > 1.5}')
            print(f'max diff: {abs(max_l-max_r)/abs(Q3_l-Q1_l) > 1.5}')   
            print('-------')
            print(f'slope left short - raw, drop na 48m [{l48_start_date.date()}:{l3_end_date.date()}] - {slope_l_short:.4f}')
            print(f'slope combined - filled 24m [{l12_start_date.date()}:{r12_end_date.date()}] - {slope_combined:.4f}')

            print(f'Large gap? {large_gap_detected}')
            print(f'Disrupted pattern? {disrupted_pattern}')
            print(f'Mean difference? {large_mean_difference}')
            print(f'[weak] Sudden quantile shift? {sudden_quantile_shift}')
            print(f'[strong] Sudden min/max_shift? {sudden_minmax_shift}')
            print(f'Sudden IQR change? {sudden_iqr_change}')
            print(f'Trend preserved? {trend_preserved}')
            print(f'-- LT gradual decline? {long_term_gradual_decline}')

        # combine all checks
        if ( large_gap_detected and disrupted_pattern and ~long_term_gradual_decline ) \
            or ( ( large_mean_difference or sudden_iqr_change ) \
                and ( ( sudden_quantile_shift or disrupted_pattern ) and ( ~trend_preserved) ) ) \
            or ( sudden_minmax_shift and ~long_term_gradual_decline ):
            filtered_breaks.append(ts_scaled.index.get_loc(b_date))

    if debug == True:
        ts_rescaled = ts_filled * (ts_max - ts_min) + ts_min
        return ts_rescaled, filtered_breaks, window_l_short, window_combined_filled, break_indices

    return ts, filtered_breaks


def estimate_slope(y_vals):
    """
    Return (slope, slope_std_err) for a simple linear regression y = a + b*x
    on consecutive points in y_vals, using x = 0, 1, 2, ...
    """
    x = np.arange(len(y_vals))
    # Add constant for intercept in statsmodels
    X = sm.add_constant(x)
    # Fit ordinary least squares
    model = sm.OLS(y_vals, X).fit()
    slope = model.params.iloc[1]
    slope_std_err = model.bse.iloc[1]
    return slope, slope_std_err


def detect_change_points(ts,pen=12):
    algo = rpt.Pelt(model="rbf").fit(ts.values)
    return algo.predict(pen=pen)[:-1]

def nan_count(arr):
        return np.count_nonzero(pd.isna(arr))
    
def find_12_window_left(series, center_date, max_nans=6):
    """
    Search outward to the LEFT for a contiguous window of EXACT length 12
    that *ends* at `center_date`. We'll allow <= max_nans missing values.
    
    center_date is an Timestamp (Series index).
    
    Returns a Series slice if found, or None if no such window is found.
    """

    center_loc = series.index.get_loc(center_date)
    window_end = center_loc
    # For radius in [0..center_loc], define 'start' as center_loc - radius.
    # We expand further left if we don't find a valid 12-length sub-window.
    for radius in range(center_loc + 1):
        left = 0
        length_sub = (window_end - center_loc + radius) + 1  # # of rows in [left..center_loc]

        if length_sub < 12:
            continue  # Not enough rows to form 12 yet, keep expanding

        # We only need to check the single exact 12-row slice that ends at center_loc:
        window_start = window_end - 11        
        if window_start < 0:
            window_start = 0

        # The sub-window is [window_start..center_loc]
        if (window_end - window_start + 1) == 12:
            window_slice = series.iloc[window_start : window_end + 1]
            if nan_count(window_slice.values) <= max_nans:
                return series.index[window_start], window_slice
        window_end -= 1

    # If we exit the loop, no 12-length window was found that ends at center_loc
    return pd.NA, pd.Series()


def find_12_window_right(series, center_date, max_nans=6):
    """
    Search outward to the RIGHT for a contiguous window of EXACT length 12
    that *starts* at `center_date`. We'll allow <= max_nans missing values.
    
    center_date is an Timestamp (Series index).
    
    Returns a Series slice if found, or None if no such window is found.
    """
    
    n = len(series)
    center_loc = series.index.get_loc(center_date)
    window_start = center_loc
    # Expand radius from 0 up to the end of the series
    for radius in range(n - center_loc):
        right = center_loc + radius
        length_sub = (right - window_start) + 1  # # of rows in [center_loc..right]
        if length_sub < 12:
            continue
        
        # For exactly 24 rows starting at center_loc:
        window_end = window_start + 11
        if window_end > right:
            window_end = right
        # Check if that sub-slice is exactly 24 long
        if (window_end - window_start + 1) == 12:
            window_slice = series.iloc[window_start : window_end + 1]
            if nan_count(window_slice.values) <= max_nans:
                return series.index[window_end], window_slice
        window_start += 1
        
    
    return pd.NA, pd.Series()

