import SimpleITK as sitk
import os
import pandas as pd

def convert_dicom_to_nrrd(dicom_folder, output_file):
    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Write the image in NRRD format
    sitk.WriteImage(image, output_file)

# Base directory containing the DICOM images
base_directory = "data_DICOM"
nrrd_directory = "nrrd_images"

# Ensure the NRRD output directory exists
os.makedirs(nrrd_directory, exist_ok=True)

# Path to the CSV file
csv_file_path = 'patient_data.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Convert each patient's DICOM to NRRD and store path
for index, row in df.iterrows():
    patient_id = row['Patient ID']
    patient_folder = os.path.join(base_directory, patient_id)
    
    t1_folder = os.path.join(patient_folder, 'T1')
    pet_folder = os.path.join(patient_folder, 'PET')
    
    if os.path.exists(t1_folder):
        t1_output_folder = os.path.join(nrrd_directory, patient_id, 'T1')
        os.makedirs(t1_output_folder, exist_ok=True)
        t1_nrrd_file = os.path.join(t1_output_folder, 'T1.nrrd')
        convert_dicom_to_nrrd(t1_folder, t1_nrrd_file)
    
    if os.path.exists(pet_folder):
        pet_output_folder = os.path.join(nrrd_directory, patient_id, 'PET')
        os.makedirs(pet_output_folder, exist_ok=True)
        pet_nrrd_file = os.path.join(pet_output_folder, 'PET.nrrd')
        convert_dicom_to_nrrd(pet_folder, pet_nrrd_file)
