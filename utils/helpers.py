from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd


# ----------------------------
# Helper: stack covariates
# ----------------------------
def stack_covariates(das_cov, months):
    """
    Returns xr.DataArray with dims ("time","lat","lon","cov") -> [T,H,W,K]
    """
    # Use first entry to get H,W, coords
    base = das_cov[0]
    H = base.sizes["lat"]; W = base.sizes["lon"]
    lat = base["lat"]; lon = base["lon"]
    T = len(months)

    chans = []
    cov_names = []
    for da in das_cov:
        name = getattr(da, "name", None) or "cov"
        cov_names.append(name)
        if "time" in da.dims:
            # select our timeline (assumes contains these times)
            sel = da.sel(time=months)
            arr = sel.to_numpy().astype("float32")  # [T,H,W]
        else:
            # static -> tile across time
            arr0 = da.to_numpy().astype("float32")  # [H,W]
            arr = np.tile(arr0[None, ...], (T, 1, 1))  # [T,H,W]
        chans.append(arr[..., None])  # [T,H,W,1]

    cov = np.concatenate(chans, axis=-1)  # [T,H,W,K]
    return xr.DataArray(
        cov,
        dims=("time","lat","lon","cov"),
        coords={"time": pd.to_datetime(months), "lat": lat, "lon": lon, "cov": cov_names},
        name="cov_stack"
    )
    


def months_between_dates(start_date_str, end_date_str):
    # Convert strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Calculate the difference in years and months
    year_diff = end_date.year - start_date.year
    month_diff = end_date.month - start_date.month

    # Total months
    total_months = year_diff * 12 + month_diff

    return total_months


def norm_xy(lat, lon, lat_min, lat_max, lon_min, lon_max):
        x = (lon - lon_min) / (lon_max - lon_min + 1e-12)  # x ~ lon
        y = (lat - lat_min) / (lat_max - lat_min + 1e-12)  # y ~ lat
        return x, y

def norm_time(ts, t0, t1):
    return (ts - t0) / (t1 - t0)

# Function to assign each point to a grid cell
def assign_to_grid(point, lats, lons):
    lat, lon = point
    lat_idx = (np.abs(lats - lat)).argmin()
    lon_idx = (np.abs(lons - lon)).argmin()
    return lat_idx, lon_idx

def sample_irregular_grid(grid: xr.DataArray, coords: np.ndarray, data: np.ndarray):
    """
    grid   (time, lat, lon)  – xarray.DataArray
    coords (n, 2)            – (lat, lon) for each station
    data   (n_time, n)       – observations to be masked
    """
    # put the station positions on a “points” dimension
    lat_da = xr.DataArray(coords[:, 0], dims="points", name="lat")
    lon_da = xr.DataArray(coords[:, 1], dims="points", name="lon")

    # broadcasted, vectorised nearest-neighbour selection
    sampled = grid.sel(lat=lat_da, lon=lon_da,
                       method="nearest")
    # apply the land mask from the model to your obs
    data_masked = data.copy()
    data_masked[np.isnan(sampled.values)] = np.nan
    return sampled.values, data_masked

# preprocess- remove leading nans
def strip_leading_nans(ts1d):
    """Remove all leading NaNs from a 1‑D NumPy array."""
    if np.all(np.isnan(ts1d)):
        raise ValueError("series is empty after stripping leading NaNs")
    first = np.argmax(~np.isnan(ts1d))
    return ts1d[first:]


def make_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean DataFrame with True only between
    first and last non-NA for each column."""
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        valid = df[col].notna()
        if valid.any():
            first_idx = valid.idxmax()
            last_idx  = valid[::-1].idxmax()
            mask.loc[first_idx:last_idx, col] = True
    return mask


# preprocess- remove leading and trialing nans
def trim_nans(s: pd.Series) -> pd.Series:
    """
    Drop leading and trailing NaNs from a pandas Series,
    while keeping NaNs between valid values.
    """
    # Find the first and last valid (non-NaN) indices
    first_valid = s.first_valid_index()
    last_valid = s.last_valid_index()

    if first_valid is None or last_valid is None:
        # Entirely NaN series → return empty Series
        return s.iloc[0:0]

    # Slice between them (inclusive)
    return s.loc[first_valid:last_valid]
