# stdk.py
# TensorFlow implementation of Space-Time.DeepKriging (Stage-1 + Stage-2)
# Compatible with the structure in https://github.com/pratiknag/Space-Time.DeepKriging
# - Stage-1: Interpolation / Imputation with spatio-temporal RBF embedding + quantile loss
# - Stage-2: Forecasting with ConvLSTM (grids), quantile outputs
# - Supports exogenous covariates provided as a full grid at each time

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Utils: quantile losses
# ----------------------------
def quantile_loss(q: float):
    q = tf.constant(q, tf.float32)
    def _loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q*e, (q-1.0)*e))
    return _loss

def multi_quantile_loss(qs: Sequence[float]):
    parts = [quantile_loss(q) for q in qs]
    def _loss(y_true, y_pred):
        # y_true: [B,1] or [B]; y_pred: [B, nQ]
        y_true = tf.reshape(y_true, (-1,1))
        return tf.add_n([p(y_true, y_pred[:,i:i+1]) for i,p in enumerate(parts)]) / len(parts)
    return _loss

def quantile_loss_masked(quantiles, lm_np):
    Q = len(quantiles)
    qs = tf.constant(quantiles, dtype=tf.float32)  # (Q,)
    # constant (H,W,1) and (H,W,Q) masks in TF
    lm_tf_1 = tf.constant(lm_np[..., None], dtype=tf.float32)        # (H,W,1)
    lm_tf_Q = tf.constant(np.repeat(lm_np[..., None], Q, axis=-1), dtype=tf.float32)  # (H,W,Q)
    def loss(y_true, y_pred):
        # y_true: (B,H,W,1), y_pred: (B,H,W,Q)
        y_true_q = tf.repeat(y_true, repeats=Q, axis=-1)          # (B,H,W,Q)
        e = y_true_q - y_pred
        q = qs[tf.newaxis, tf.newaxis, tf.newaxis, :]             # (1,1,1,Q)
        pinball = tf.maximum(q*e, (q-1.0)*e)                      # (B,H,W,Q)
        # mask ocean out
        masked = pinball * lm_tf_Q
        denom = tf.reduce_sum(lm_tf_Q) + tf.keras.backend.epsilon()
        return tf.reduce_sum(masked) / denom
    return loss

# ----------------------------
# Utils: RBF embedding
# ----------------------------
def _pairwise_sq_dists(X, C):
    # X: [N,D], C: [M,D]
    X2 = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)
    C2 = tf.reduce_sum(tf.square(C), axis=1, keepdims=True)
    XC = tf.matmul(X, C, transpose_b=True)
    return X2 - 2*XC + tf.transpose(C2)

class RBFEmbed(layers.Layer):
    def __init__(self, centers: np.ndarray, lengthscale: float, name="RBFEmbed", trainable_centers=False, trainable_ls=False):
        super().__init__(name=name)
        self.C = tf.Variable(tf.convert_to_tensor(centers, tf.float32), trainable=trainable_centers)
        self.log_ls = tf.Variable(tf.math.log(tf.constant(lengthscale, tf.float32)), trainable=trainable_ls)

    def call(self, X):
        # X: [B,D]
        d2 = _pairwise_sq_dists(tf.cast(X, tf.float32), self.C)
        ls2 = tf.square(tf.exp(self.log_ls)) + 1e-9
        return tf.exp(-0.5 * d2 / ls2)

# -----------------------------------------------------------
# Stage-1: Interpolator (point-wise)
# Inputs:
#   coords_xy: [B,2] in [0,1]^2
#   time_scalar:  [B,1] in [0,1]
#   cov_scalar_vec:  [B,Nc] covariates sampled at coords
# Output:
#   y-quantiles: [B, nQ]
# ----------------------------------------------------------

def build_interpolator_scalar_cov(
    n_space_basis=64, n_time_basis=16, ls_space=0.15, ls_time=0.10,
    hidden=(256,128,64), dropout=0.1, quantiles=(0.1,0.5,0.9),
    space_centers=None, time_centers=None, n_scalar_cov: int = 1
):
    # centers (grid if none provided)
    if space_centers is None:
        g = int(np.ceil(np.sqrt(n_space_basis)))
        xs = np.linspace(0,1,g); X,Y = np.meshgrid(xs,xs)
        space_centers = np.stack([X.ravel(), Y.ravel()], axis=1)
    if time_centers is None:
        time_centers = np.linspace(0,1,n_time_basis)[:,None]

    inp_xy = keras.Input(shape=(2,),  name="coords_xy")
    inp_t  = keras.Input(shape=(1,),  name="time_scalar")
    inp_c  = keras.Input(shape=(n_scalar_cov,),  name="cov_scalar_vec")  # K scalars

    phi_s = RBFEmbed(space_centers, ls_space, name="phi_space")(inp_xy)
    phi_t = RBFEmbed(time_centers,  ls_time,  name="phi_time")(inp_t)
    x = layers.Concatenate()([phi_s, phi_t, inp_c])

    for i,h in enumerate(hidden):
        x = layers.Dense(h, activation="relu", name=f"dnn_{i}")(x)
        x = layers.Dropout(dropout)(x)

    out = layers.Dense(len(quantiles), name="q_out")(x)

    model = keras.Model(inputs=[inp_xy, inp_t, inp_c], outputs=out, name="STDK_Interp_ScalarCov")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=multi_quantile_loss(quantiles))
    return model

# ------------------------------------------------------
# Interpolator helper: run interpolation for all months
# ------------------------------------------------------
def predict_all_months_scalar_cov(model_i, das_cov, months, coords_xy01_grid, t0, t1):
    H = das_cov[0].sizes["lat"]; W = das_cov[0].sizes["lon"]
    nQ = model_i.output_shape[-1]
    out = np.empty((len(months), H, W, nQ), dtype="float32")
    for k, ts in enumerate(pd.to_datetime(months)):
        out[k] = predict_grid_scalar_cov(model_i, das_cov, ts, coords_xy01_grid, t0, t1, H, W)
    quant_names = [f"{q}" for q in getattr(model_i, "quantiles", (0.1,0.5,0.9))]  # fallback names
    return xr.DataArray(
        out,
        dims=("time","lat","lon","quantile"),
        coords={"time": pd.to_datetime(months), "lat": das_cov[0]["lat"], "lon": das_cov[0]["lon"],
                "quantile": quant_names}
    )

def predict_grid_scalar_cov(model_i, das_cov, ts, coords_xy01_grid, t0, t1, H, W):
    # ---- 1) normalize time to [0,1] ----
    t_scalar = ((ts - t0) / (t1 - t0))
    t_scalar = np.clip(t_scalar, 0.0, 1.0).astype("float32")
    t_scalar = np.full((H * W, 1), t_scalar, dtype="float32")  # broadcast for all grid pts

    # ---- 2) prepare scalar covariates ----
    cov_vecs = []
    for da in das_cov:
        # select covariate slice at timestamp (using nearest time)
        if da.name == 'dem':
            cov2d = da.values.astype("float32")
        else:
            cov2d = da.sel(time=ts, method="nearest").values.astype("float32")
        cov_vecs.append(cov2d.reshape(-1))  # flatten (H*W,)

    cov_stack = np.stack(cov_vecs, axis=1)  # (H*W, n_cov)

    # ---- 3) predict all grid points ----
    y_q = model_i.predict(
        {"coords_xy": coords_xy01_grid,
         "time_scalar": t_scalar,
         "cov_scalar_vec": cov_stack},
        verbose=0,
        batch_size=4096,
    )  # (H*W, nQ)

    # ---- 4) reshape back to grid ----
    nQ = y_q.shape[-1]
    out_q = y_q.reshape(H, W, nQ).astype("float32")

    return out_q

# ------------------------------------------------------------------------------------------
# Stage-2: Forecasting
#   ConvLSTM for gridded forecasting (uses full grid covariate sequences), outputs quantiles
# ------------------------------------------------------------------------------------------

def build_grid_forecaster(input_shape, quantiles, land_mask, hidden=(32, 32)) -> keras.Model:
    Q = len(quantiles)
    inp = keras.Input(shape=input_shape)
    x = inp
    for h in hidden[:-1]:
        x = layers.ConvLSTM2D(h, 3, padding='same', return_sequences=True, activation='tanh')(x)
    x = layers.ConvLSTM2D(hidden[-1], 3, padding='same', return_sequences=False, activation='tanh')(x)
    out = layers.Conv2D(Q, 1, padding='same', activation=None)(x)  # (B,H,W,Q)
    model = keras.Model(inp, out, name="STDK_ConvLSTM_Quantiles_TF_Masked")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=quantile_loss_masked(quantiles, land_mask))
    return model

# ----------------------------------------------------------
# Forecaster helper: run forecaster for n steps recursively
# ----------------------------------------------------------
def multi_step_forecast(
    model, X_all, steps, in_len, lm_np, q_list, median_q=0.5, time_ref=None):

    Q = len(q_list)
    mid = q_list.index(median_q)
    ctx = X_all[-in_len:].copy()  # (in_len, H, W, C)
    outs = []

    for _ in range(steps):
        # predict quantiles for next frame
        pred_q = model.predict(ctx[np.newaxis, ...], verbose=0)[0]  # (H, W, Q)
        # enforce ocean = -1 (normalized space)
        pred_q = np.where(lm_np[..., None], pred_q, -1.0)
        outs.append(pred_q)

        # build next context frame (1, H, W, C)
        next_frame = np.zeros_like(ctx[0:1])  # (1, H, W, C)
        med_norm = pred_q[..., mid]           # (H, W)
        next_frame[..., 0] = np.where(lm_np, med_norm, -1.0)
        if ctx.shape[-1] > 1:
            next_frame[..., 1:] = ctx[-1:, ..., 1:]  # keep exog

        # roll the context window
        ctx = np.concatenate([ctx[1:], next_frame], axis=0)

    pred_seq = np.stack(outs, axis=0)  # (steps, H, W, Q)

    # ---- build monthly timestamps ----
    if time_ref is not None:
        last_time = pd.to_datetime(time_ref[-1])
        forecast_times = pd.date_range(
            start=last_time + pd.offsets.MonthBegin(1),
            periods=steps,
            freq="MS"
        ) + pd.offsets.Day(14)  # move to 15th
    else:
        forecast_times = pd.date_range("2000-01-15", periods=steps, freq="MS")  # dummy

    return pred_seq, forecast_times



