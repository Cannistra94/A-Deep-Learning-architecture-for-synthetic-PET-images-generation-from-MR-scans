# Initialize KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
fold_idx = 0
reduce_lr = ReduceLROnPlateau(monitor='val_psnr_metric', factor=0.2, patience=10, min_lr=1e-7, verbose=1)
#early_stopping = EarlyStopping(monitor='val_psnr_metric', patience=8, restore_best_weights=True, verbose=1)
# Create directories to save predictions as images
save_dir = 'december21_fullimage/prediction_images_10fold'
save_dir_nii = 'december21_fullimage/prediction_images_nii_10fold'
output_dir = 'december21_fullimage/output_plots'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir_nii, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
# List to store metrics across folds
mae_scores = []
mse_scores = []
psnr_scores = []
ssim_scores = []

# Lists to store training and validation metrics
train_loss_per_fold = []
val_loss_per_fold = []
train_psnr_per_fold = []
val_psnr_per_fold = []

for train_idx, test_idx in kfold.split(X_slices):
    fold_idx += 1
    print(f"Fold {fold_idx}")

    # Split data into training and test sets
    X_train, X_test = X_slices[train_idx], X_slices[test_idx]
    y_train, y_test = y_slices[train_idx], y_slices[test_idx]
    test_patient_labels = patient_labels[test_idx]  # Get the test patient labels
    # Split training set further into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    # Reshape data to add the channel dimension if necessary
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    

    print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples, testing on {X_test.shape[0]} samples")

    # Build the 2D U-Net model
    input_shape = X_train.shape[1:]
    
    model = resnet_unet_2d_1mm_new(input_shape)

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=32, verbose=2, callbacks=[reduce_lr])
    # Save training and validation metrics
    train_loss_per_fold.append(history.history['loss'])
    val_loss_per_fold.append(history.history['val_loss'])
    train_psnr_per_fold.append(history.history['psnr_metric'])
    val_psnr_per_fold.append(history.history['val_psnr_metric'])
    # Evaluate the model on test data
    y_pred_test = model.predict(X_test)

    # Compute metrics for this fold on test set
    mae_test, mse_test, psnr_test, ssim_test = compute_metrics(y_test, y_pred_test)
    mae_scores.append(mae_test)
    mse_scores.append(mse_test)
    psnr_scores.append(psnr_test)
    ssim_scores.append(ssim_test)
    # Plot and save training and validation loss and PSNR for the current fold
    plt.figure(figsize=(12, 6))

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold_idx} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Training and Validation PSNR
    plt.subplot(1, 2, 2)
    plt.plot(history.history['psnr_metric'], label='Training PSNR')
    plt.plot(history.history['val_psnr_metric'], label='Validation PSNR')
    plt.title(f'Fold {fold_idx} - Training and Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fold_{fold_idx}_metrics_duringtraining.png'))
    plt.close()
    
    # Save predictions and ground truths as images
    for i in range(len(y_pred_test)):
        patient_id = test_patient_labels[i]
        slice_index = test_idx[i] % X.shape[3]  # Calculate slice index without exclusions

                y_pred_img = (y_pred_test[i] * 255).astype(np.uint8)
        y_test_img = (y_test[i] * 255).astype(np.uint8)
        x_test_img = (X_test[i] * 255).astype(np.uint8)

        # Save images 
        cv2.imwrite(os.path.join(save_dir, f"fold_{fold_idx}_patient_{patient_id}_slice_{slice_index}_y_pred.png"), y_pred_img)
        cv2.imwrite(os.path.join(save_dir, f"fold_{fold_idx}_patient_{patient_id}_slice_{slice_index}_y_test.png"), y_test_img)
        cv2.imwrite(os.path.join(save_dir, f"fold_{fold_idx}_patient_{patient_id}_slice_{slice_index}_X_test.png"), x_test_img)

        # Save images in .nii format
        y_pred_nii = nib.Nifti1Image(y_pred_test[i], affine=np.eye(4))
        y_test_nii = nib.Nifti1Image(y_test[i], affine=np.eye(4))
        x_test_nii = nib.Nifti1Image(X_test[i], affine=np.eye(4))

        nib.save(y_pred_nii, os.path.join(save_dir_nii, f"fold_{fold_idx}_patient_{patient_id}_slice_{slice_index}_y_pred.nii"))
        nib.save(y_test_nii, os.path.join(save_dir_nii, f"fold_{fold_idx}_patient_{patient_id}_slice_{slice_index}_y_test.nii"))
        nib.save(x_test_nii, os.path.join(save_dir_nii, f"fold_{fold_idx}_patient_{patient_id}_slice_{slice_index}_X_test.nii"))

        # Print metrics for the current fold on test set
        print(f"Test set metrics - Fold {fold_idx}: MAE: {mae_test}, MSE: {mse_test}, PSNR: {psnr_test}, SSIM: {ssim_test}")
