#!/bin/bash

# read inputs

list=$1

dos2unix $list

for filename in `cat $list`; do

antsBrainExtraction.sh -d 3 -a resampled/${filename}_T1_resampled.nii -e MNI152_T1_1mm.nii.gz -m MNI152_T1_1mm_brain_mask.nii.gz -o skull_stripped_1mm/${filename}_T1 -s .nii 

done
