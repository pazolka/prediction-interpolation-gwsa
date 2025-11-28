import numpy as np
import xarray as xr
import tensorflow as tf

from sklearn.metrics import r2_score, mean_squared_error
from utils.helpers import sample_irregular_grid

def global_MSE(data1, data2):
    return round(mean_squared_error(data1, data2), 4)

def global_R2(data1, data2):
    return round(r2_score(data1, data2), 2)

# calculate global MSE
# point to grid
def xr_global_point_MSE(point_values, point_coords, grid):
    lats, lons = point_coords[:,0], point_coords[:,1]
    sampled_values = grid.sel(lat=xr.DataArray(lats, dims="points"),
                                        lon=xr.DataArray(lons, dims="points"),
                                        method="nearest")
    if np.isnan(sampled_values.values).any(): # workaround for issue with 0.05 grid
        sampled_values = grid.sel(lat=xr.DataArray(lats, dims="points"),
                                    lon=xr.DataArray(lons, dims="points"),
                                    method="ffill")

    sampled_values = sampled_values.values
    obs_nan_mask = ~np.isnan(point_values.ravel()) & ~np.isnan(sampled_values.ravel())
    return global_MSE(point_values.ravel()[obs_nan_mask], sampled_values.ravel()[obs_nan_mask])

# calculate global R2
# point to grid
def xr_global_point_R2(point_values, point_coords, grid):
    lats, lons = point_coords[:,0], point_coords[:,1]
    sampled_values = grid.sel(lat=xr.DataArray(lats, dims="points"),
                                        lon=xr.DataArray(lons, dims="points"),
                                        method="nearest")
    if np.isnan(sampled_values.values).any(): # workaround for issue with 0.05 grid
        sampled_values = grid.sel(lat=xr.DataArray(lats, dims="points"),
                                        lon=xr.DataArray(lons, dims="points"),
                                        method="ffill")
    sampled_values = sampled_values.values
    obs_nan_mask = ~np.isnan(point_values.ravel()) & ~np.isnan(sampled_values.ravel())
    return global_R2(point_values.ravel()[obs_nan_mask], sampled_values.ravel()[obs_nan_mask])

# calculate global R2
# point to point
def calculate_global_R2(data1, data2):
    flattened1 = data1.ravel()
    mask = ~np.isnan(flattened1)
    filtered1 = flattened1[mask]
    
    flattened2 = data2.ravel()
    filtered2 = flattened2[mask]

    if len(filtered2) <= 1:
        return np.nan
    return global_R2(filtered1, filtered2)

# calculate global MSE
# point to point
def calculate_global_MSE(data1, data2):
    flattened1 = data1.ravel()
    mask = ~np.isnan(flattened1)
    filtered1 = flattened1[mask]
    
    flattened2 = data2.ravel()
    filtered2 = flattened2[mask]

    if len(filtered2) <= 1:
        return np.nan
    return global_MSE(filtered1, filtered2)
