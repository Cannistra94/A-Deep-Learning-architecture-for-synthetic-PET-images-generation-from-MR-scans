#functions are defined for metrics computation
def compute_metrics(y_true, y_pred):
    # Ensure y_pred has the same shape as y_true
    if y_true.shape != y_pred.shape:
        y_pred = np.squeeze(y_pred)  # Remove the extra dimension if necessary
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
  
    y_true_tf = tf.image.convert_image_dtype(y_true, tf.float64)
    y_pred_tf = tf.image.convert_image_dtype(y_pred, tf.float64)
    
    psnr = tf.image.psnr(y_true_tf, y_pred_tf, max_val=1.0)
    ssim = tf.image.ssim(y_true_tf, y_pred_tf, max_val=1.0)
    
    return mae, mse, psnr, ssim

def psnr_metric(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        y_pred = tf.squeeze(y_pred, axis=-1)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    # Ensure y_pred has the same shape as y_true
    y_pred = tf.squeeze(y_pred, axis=-1)  # Remove the last channel dimension from y_pred
    return tf.image.ssim(y_true, y_pred, max_val=1.0)
