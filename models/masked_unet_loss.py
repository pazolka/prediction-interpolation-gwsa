import tensorflow as tf

def _resize_bool_mask(mask):
    # mask: [H0,W0] bool
    mask = tf.convert_to_tensor(mask)
    if mask.shape.rank == 2:
        mask = mask[..., tf.newaxis]     # [H0,W0,1]
    if mask.shape.rank == 3:
        mask = mask[tf.newaxis, ...]     # [1,H0,W0,1]
    return mask  # keep it binary

def make_masked_mse_loss(fixed_mask_hw):
    """Return a loss(y_true,y_pred) that averages MSE over True pixels only."""
    mask = _resize_bool_mask(fixed_mask_hw)        # [1,H,W,1], bool
    mask_f = tf.cast(mask, tf.float32)

    @tf.function
    def loss_fn(y_true, y_pred):
        # per-pixel MSE across channels
        per_pix = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true), axis=-1, keepdims=True)  # [B,H,W,1]
        per_pix = tf.where(mask, per_pix, 0.0)
        denom = tf.reduce_sum(mask_f)  # number of valid pixels (same every step)
        return tf.reduce_sum(per_pix) / tf.maximum(denom, 1e-12)
    return loss_fn


def make_masked_mse_with_trend(
    fixed_mask_hw,
    w_main=1.0,     # weight of standard masked MSE
    w_trend=0.3,    # weight of low-frequency (trend) loss
    w_mean=0.1,     # weight of domain mean loss
    ksize=9         # low-pass kernel size
):
    """
    Returns loss(y_true, y_pred) = w_main*MSE_masked + w_trend*MSE_lowpass_masked + w_mean*MSE_mean_masked.
    - fixed_mask_hw: [H, W] or [H, W, 1] bool array indicating valid pixels.
    - Low-pass uses normalized average pooling to respect missing data.
    """

    mask = _resize_bool_mask(fixed_mask_hw)            # [1, H, W, 1], bool
    mask_f = tf.cast(mask, tf.float32)                 # same shape
    denom_spatial = tf.reduce_sum(mask_f)              # scalar (constant across steps)

    k = int(ksize)
    if k % 2 == 0:
        raise ValueError("ksize must be odd")

    def lowpass_normalized(x):
        """
        x: [B, H, W, 1]
        Returns normalized avg_pool2d(x * mask) / avg_pool2d(mask), avoiding bias near gaps.
        """
        num = tf.nn.avg_pool2d(x * mask_f, ksize=k, strides=1, padding='SAME')
        den = tf.nn.avg_pool2d(mask_f,     ksize=k, strides=1, padding='SAME')
        return num / tf.maximum(den, 1e-6)

    @tf.function
    def loss_fn(y_true, y_pred):
        # ---- 1) Standard masked per-pixel MSE ----
        per_pix = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true), axis=-1, keepdims=True)  # [B,H,W,1]
        per_pix = per_pix * mask_f
        main_loss = tf.reduce_sum(per_pix) / tf.maximum(denom_spatial, 1e-12)

        # ---- 2) Low-frequency (trend) masked MSE ----
        y_true_lp = lowpass_normalized(y_true)   # [B,H,W,1]
        y_pred_lp = lowpass_normalized(y_pred)
        lp_err = tf.math.squared_difference(y_pred_lp, y_true_lp) * mask_f
        trend_loss = tf.reduce_sum(lp_err) / tf.maximum(denom_spatial, 1e-12)

        # ---- 3) Masked spatial-mean MSE ----
        # compute per-batch means over valid pixels
        sum_true = tf.reduce_sum(y_true * mask_f, axis=[1, 2, 3], keepdims=True)  # [B,1,1,1]
        sum_pred = tf.reduce_sum(y_pred * mask_f, axis=[1, 2, 3], keepdims=True)
        mean_true = sum_true / tf.maximum(denom_spatial, 1e-12)
        mean_pred = sum_pred / tf.maximum(denom_spatial, 1e-12)
        mean_loss = tf.reduce_mean(tf.square(mean_pred - mean_true))  # average over batch

        return w_main * main_loss + w_trend * trend_loss + w_mean * mean_loss

    return loss_fn
