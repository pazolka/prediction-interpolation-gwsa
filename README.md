# Predicting and Interpolating Spatiotemporal Environmental Data: A Case Study of Groundwater Storage in Bangladesh
Paper submitted to the [IDA 2026](https://ida2026.liacs.nl/) conference

**Goals**: evaluating several deep learning models in estimating a spatiotemporal target; extrapolating from point observations to a continuous field.

**Target**: groundwater storage anomalies (GWSA) wrt 2004-2009

**Tested models**:
- 2D Unet
- 3D Unit
- 3D DeepKriging [1]
- CNN or CNN+LSTM stack
- CNN or CNN+LSTM stack + 2D Kriging
- GLDAS 2.2 CLSM GWS (monthly average) - as baseline [2]

**Input variables**:
- MERIT DEM [3]
- CSR GRACE TWSA [4,5]
- TerraClimate Precipitation and Actual Evaportanspiration [6]
- MODIS NDVI [7]

## Input and output data

**Gridded model inputs**: raw (_.netcdf_) and processed (_.npy_) can be found in `data/Bangladesh/inputs`

**Gridded model outputs**: _.netcdf_ files can be found in `data/Bangladesh/outputs/{model}`

**Point model outputs**: _.npy files can be found in `data/Bangladesh/outputs/g2p_CNN or g2p_CNN_LSTM`. Their corresponding IDs and lat/lon coordinates are in `data/Bangladesh/target/predicted_point_coords_ids.npy`.

Raw target dataset (original weekly monitored groundwater level time-series dataset) is unpublished and can be obtained from the [Bangladesh Water Development Board](http://www.hydrology.bwdb.gov.bd/index.php) though an online data request and payment. Processed monthly time-series dataset used in this work was compiled by [8].

Missing target files:
- `data/Bangladesh/target/raw_data.csv`
- `data/Bangladesh/target/filtered_gws_ts_data_1961_2019.csv`
- `data/Bangladesh/target/filtered_filled_missForest_gws_ts_data_1961_2019.csv`

Missing input files (due to a large size):
- `data/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc` - CSR GRACE RL06 Mascons terrestrial water storage changes [4,5]
- `data/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc` - CSR GRACE RL06 Mascons land mask
  
Both missing files can be downloaded [here](https://www2.csr.utexas.edu/grace/RL06_mascons.html). 

## Repository structure:
Jupyter notebooks and an R script contain the complete code for reproducing the analysis based on 10-fold cross-validation.

- `0_cv_experiments.ipynb` initial experiment setup & model evaluation.
- `1_data_preparation.ipynb` preparation of the gridded inputs.
- `2_1_target_data_cleanup.ipynb` preparation of the point target - note that this involves time series preprocessing based on rules specified in `utils/changepoint_helpers.py` and `utils/outliers_helpers.py`, as well as manual expert-led visual examination of time series graphs.
- `2_2_data_imputation_missForest.R` temporal gap filling based on _missForest_ [9].
- `3_2d_unet_cv.ipynb` Gridded target generation based on the 2D (spatial only) Unet as defined in `models/grid_to_grid.py#unet2d`.
- `4_3d_unet_cv.ipynb` Gridded target generation based on the 3D (space-time) Unet as defined in `models/grid_to_grid.py#unet3d`.
- `5_g2p_cnn_kriging_cv.ipynb` Point target generation based on the 2D (spatial only) CNN with a subsequent 2D Ordinary Kriging step for a gridded version; g2p model defined in `models/grid_to_point.py#cnn`.
- `6_g2p_cnn_lstm_kriging_cv.ipynb` Point target generation based on the 3D (space-time) CNN-LSTM with a subsequent 2D Kriging step; g2p model defined in `models/grid_to_point.py#cnn_lstm`.

## Cross-validation setup
To test both prediction and interpolation, a holdout set was set aside at the
start, consisting of 8% of the available locations selected randomly. Next, 10-fold
temporal cross validation was performed on the remaining data, splitting
the time series along the time dimension into train, validation and test sets, with a progressively growing
training set, and validation and test sets of fixed length of 8 time steps

The setup used for generating the published outputs can be found in `data/Bangladesh/cv_setup.npy`.

#### References:

[1] Nag, P., Sun, Y., Reich, B.J (2023), Spatio-temporal DeepKriging for interpolation and probabilistic forecasting. Spatial Statistics 57, 100773, doi:10.1016/j.spasta.2023.100773.

[2] Li, B. et al. (2019), Global GRACE Data Assimilation for Groundwater and Drought Monitoring: Advances and Challenges. Water Resources Research 55(9), 7564–7586, doi:10.1029/2018WR024618

[3] Yamazaki D., D. Ikeshima, R. Tawatari, T. Yamaguchi, F. O'Loughlin, J.C. Neal, C.C. Sampson, S. Kanae & P.D. Bates. A high accuracy map of global terrain elevations. Geophysical Research Letters, vol.44, pp.5844-5853, 2017, doi:10.1002/2017GL072874

[4] Save, H., S. Bettadpur, and B.D. Tapley (2016), High resolution CSR GRACE RL05 mascons, J. Geophys. Res. Solid Earth, 121, doi:10.1002/2016JB013007.

[5] Save, Himanshu, 2020, "CSR GRACE and GRACE-FO RL06 Mascon Solutions v02", doi: 10.15781/cgq9-nh24.

[6] Abatzoglou, J.T., S.Z. Dobrowski, S.A. Parks, K.C. Hegewisch, 2018, Terraclimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958-2015, Scientific Data 5:170191, doi:10.1038/sdata.2017.191

[7] Didan, Kamel, MODIS/Terra Vegetation Indices Monthly L3 Global 1km SIN Grid V061. NASA Land Processes Distributed Active Archive Center, 2021, doi:10.5067/MODIS/MOD13A3.061. Date Accessed: 2025-11-28

[8] Mohammad Shamsudduha et al. (2022), The Bengal Water Machine: Quantified freshwater capture in Bangladesh. Science 377,1315-1319, doi:10.1126/science.abm4730.

[9] Stekhoven DJ, Bühlmann P (2012), MissForest: nonparametric missing value imputation for mixed-type data. Bioinformatics, 28(1), 112–118. doi:10.1093/bioinformatics/btr597.
