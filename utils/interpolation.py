from scipy.interpolate import griddata
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Simple gridded interpolation using scipy's griddata
def griddata_interpolation(x,y,var):
    lon_grid, lat_grid = np.meshgrid(x, y)
    points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
    values = var.ravel()

    # Mask for missing values
    mask = np.isnan(values)
    points_known = points[~mask]
    values_known = values[~mask]
    
    # Missing points
    points_missing = points[mask]
    
    # Define a grid for interpolation
    grid_x, grid_y = np.meshgrid(x, y)

    # Perform interpolation
    interpolated_values = griddata(points_known, values_known, points_missing, method='cubic')
    
    # Place interpolated values back into the original array
    values_filled = values.copy()
    values_filled[mask] = interpolated_values
    
    # Reshape back to the original grid
    filled_var = values_filled.reshape(var.shape)
    return filled_var

