#!/bin/bash

# read inputs

list=$1

dos2unix $list

for filename in `cat $list`; do

 # Perform registration of PET to MR
antsRegistrationSyNQuick.sh -d 3 -f skull_stripped_1mm/${filename}_T1BrainExtractionBrain..nii -m skull_stripped_1mm/${filename}_PETBrainExtractionBrain..nii -o rm_pet_coregistered/${filename}_rigid

# Apply the transformation to the PET image
antsApplyTransforms -d 3 -i skull_stripped_1mm/${filename}_T1BrainExtractionBrain..nii -r skull_stripped_1mm/${filename}_PETBrainExtractionBrain..nii -t [rm_pet_coregistered/${filename}_rigid0GenericAffine.mat, 1] -t rm_pet_coregistered/${filename}_rigid1Warp.nii.gz -o rm_pet_coregistered/${filename}_registered.nii.gz

done
