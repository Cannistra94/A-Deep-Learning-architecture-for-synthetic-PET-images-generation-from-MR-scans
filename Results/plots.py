# Plotting test metrics across folds
plt.figure(figsize=(12, 6))

# Plot MAE across folds
plt.subplot(2, 2, 1)
plt.plot(range(1, 11), mae_scores, label='MAE', marker='o')
plt.title('MAE Across Folds')
plt.xlabel('Fold')
plt.ylabel('MAE')
plt.grid(True)

# Plot MSE across folds
plt.subplot(2, 2, 2)
plt.plot(range(1, 11), mse_scores, label='MSE', marker='o', color='r')
plt.title('MSE Across Folds')
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.grid(True)

# Plot PSNR across folds
plt.subplot(2, 2, 3)
plt.plot(range(1, 11), psnr_scores, label='PSNR', marker='o', color='g')
plt.title('PSNR Across Folds')
plt.xlabel('Fold')
plt.ylabel('PSNR')
plt.grid(True)

# Plot SSIM across folds
plt.subplot(2, 2, 4)
plt.plot(range(1, 11), ssim_scores, label='SSIM', marker='o', color='m')
plt.title('SSIM Across Folds')
plt.xlabel('Fold')
plt.ylabel('SSIM')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'test_metrics_across_folds.png'))
plt.close()
