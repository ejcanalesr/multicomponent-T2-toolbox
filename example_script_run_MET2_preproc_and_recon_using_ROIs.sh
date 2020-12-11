#!/bin/bash 

Data=MET2.nii.gz
ROIs=ROI_image.nii.gz
# ROIs is a 3D image with different regions of interest, i.e. fiber bundles/tracts registered to the native anatomical space.
# All voxels belonging to the same ROI must be labeled with the same integer number, and different numbers must be used for different ROIs.
Path='/media/Disco1T/images'

reg_matrix="I"
#choices="I", "L1", "L2", "InvT2"

list_of_subjects="
subject_1
subject_2
"

mkdir Recon_folder

#Note that: Path_to_Data=$Path/$subi/$Data
for subi in $list_of_subjects; do
   echo "=========================== Subject:   " $subi " ==========================="
   Path_to_Data=$Path/$subi/$Data
   Path_to_ROIs=$Path/$subi/$ROIs
   
   # Check if the subject's data exists"
   if [ -f $Path_to_Data ]
   then
       #_____________ Run the preprocessing and reconstruction _____________#
       # Create list of processed subjects
       echo $subi >> Recon_folder/computed_subjects.txt

       mkdir Recon_folder/$subi

       # Copy data to local folder (using FSL, otherwise use cp)
       echo "(1) Copy data to local folder"
       fslmaths $Path_to_Data  Recon_folder/$subi/Data.nii.gz
       fslmaths $Path_to_ROIs  Recon_folder/$subi/ROIs.nii.gz

       # Remove Gibbs Ringing Artifacts using MRtrix3 (this step is optional)
       echo "(2) Remove Gibbs Ringing Artifacts, please wait..."
       mrdegibbs Recon_folder/$subi/Data.nii.gz Recon_folder/$subi/Data.nii.gz -force
   
       # Brain extraction (BET) using FSL
       echo "(3) BET, please wait..."
       fslmaths Recon_folder/$subi/Data.nii.gz -Tmean Recon_folder/$subi/Data_avg.nii.gz
       bet Recon_folder/$subi/Data_avg.nii.gz Recon_folder/$subi/Data_mask -m -v -f 0.4
       mv Recon_folder/$subi/Data_mask_mask.nii.gz Recon_folder/$subi/mask.nii.gz
       
       # Estimate T2 spectra
       # This variant will compute a single T2 distribution per ROI for a set of predefined ROIs (by computing the average of all signals for each ROI and by taking into account the distribution of flip angles within the ROI)
       echo "(4) ROI-based non-parametric T2 estimation, please wait..."
       python run_real_data_script_ROI_based_estimation.py --path_to_folder='Recon_folder'/$subi/ --input='Data.nii.gz' --mask='mask.nii.gz' --ROIs='ROIs.nii.gz' --minTE=10.68 --nTE=32 --TR=1000  --FA_method='spline' --FA_smooth='yes' --denoise='TV'  --reg_matrix=$reg_matrix --myelin_T2=40 --numcores=-1
       # Optionally, you can disable the denosing by using --denoise='None', and the data can be denoised with any external software
       # The values: minTE, nTE, and TR, must be modified according to the parameters used in the scanner.
   else
       echo "Error: Data " $Path_to_Data " does not exist"
       echo $subi >> Recon_folder/subjects_with_problems.txt
   fi
done

