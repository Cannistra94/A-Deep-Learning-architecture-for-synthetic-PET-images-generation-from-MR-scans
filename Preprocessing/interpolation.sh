#!/bin/bash

# read inputs

list=$1

dos2unix $list

for filename in `cat $list`; do

ResampleImage 3 n4_corrected/${filename}_PET.nii ${filename}_PET.nii 1x1x1 3

echo "Resampled n4_corrected/${filename}_PET.nii to resampled/${filename}_T1.nii with 1x1x1 spacing using BSpline interpolation"

 

done
