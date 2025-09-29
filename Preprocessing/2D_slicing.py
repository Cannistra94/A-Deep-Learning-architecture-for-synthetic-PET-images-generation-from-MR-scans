import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to normalize image data to range 0-1
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Path to the folder 
folder_path = 'flirt_1mm'
csv_path ='patient_data.csv' 

# Load the CSV file
df = pd.read_csv(csv_path)

# Initialize lists to store the images
X = []
y = []


for index, row in df.iterrows():
    subject_id = row['Patient ID']
    
    # Construct the filenames
    t1_filename = os.path.join(folder_path, f"{subject_id}_T1.nii.gz")
    pet_filename = os.path.join(folder_path, f"{subject_id}_PET.nii.gz")
    
    # Load the T1 and PET images
    t1_img = nib.load(t1_filename)
    pet_img = nib.load(pet_filename)
    
    # Get the image data as numpy arrays
    t1_data = t1_img.get_fdata()
    pet_data = pet_img.get_fdata()
    
    # Append the data to the lists
    X.append(t1_data)
    y.append(pet_data)

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} T1 images and {len(y)} PET images.")


#AXIAL SLICES

X_slices = []
y_slices = []

for i in range(X.shape[0]):
    for j in range(25, X.shape[3] - 25):  
        X_slices.append(X[i, :, :, j])
        y_slices.append(y[i, :, :, j])

# Convert lists to numpy arrays
X_slices = np.array(X_slices)
y_slices = np.array(y_slices)

print(f"X_slices shape: {X_slices.shape}") 
print(f"y_slices shape: {y_slices.shape}")

# Print some examples to verify matching PET-MR
num_examples = 3  

for i in range(num_examples):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.title(f"X_slice {i}")
    plt.imshow(X_slices[i], cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title(f"y_slice {i}")
    plt.imshow(y_slices[i], cmap='gray')
    
    plt.show()


X_slices = []
y_slices = []
patient_labels = []  # To keep track of the patient index

for i in range(X.shape[0]):
    for j in range(X.shape[3]):  
        X_slices.append(X[i, :, :, j])
        y_slices.append(y[i, :, :, j])
        patient_labels.append(i)  # Append patient index

# Convert lists to numpy arrays
X_slices = np.array(X_slices)
y_slices = np.array(y_slices)
patient_labels = np.array(patient_labels)

print(f"X_slices shape: {X_slices.shape}")  
print(f"y_slices shape: {y_slices.shape}")
print(f"patient_labels shape: {patient_labels.shape}")
