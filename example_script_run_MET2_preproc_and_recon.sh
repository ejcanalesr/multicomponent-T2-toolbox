#!/bin/bash 

Data=MET2.nii.gz
Path='/media/Disco1T/images'

list_of_methods="
X2
L_curve
"
# choices="NNLS", "T2SPARC", "X2", "L_curve", "GCV", "BayesReg"

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
   # Check if the subject's data exists"
   if [ -f $Path_to_Data ]  
   then
       #_____________ Run the preprocessing and reconstruction _____________#
       # Create list of processed subjects
       echo $subi >> Recon_folder/computed_subjects.txt

       mkdir Recon_folder
       mkdir Recon_folder/$subi

       # Copy data to local folder (using FSL, otherwise use cp)
       echo "(1) Copy data to local folder"
       fslmaths $Path_to_Data      Recon_folder/$subi/Data.nii.gz
       
       # Remove Gibbs Ringing Artifacts using MRtrix3 (this step is optional)
       echo "(2) Remove Gibbs Ringing Artifacts, please wait..."
       mrdegibbs Recon_folder/$subi/Data.nii.gz Recon_folder/$subi/Data.nii.gz -force
       
       # Brain extraction (BET) using FSL
       echo "(3) BET, please wait..."
       fslmaths Recon_folder/$subi/Data.nii.gz -Tmean Recon_folder/$subi/Data_avg.nii.gz
       bet Recon_folder/$subi/Data_avg.nii.gz Recon_folder/$subi/Data_mask -m -v -f 0.4
       mv Recon_folder/$subi/Data_mask_mask.nii.gz Recon_folder/$subi/mask.nii.gz
       
       # Estimate T2 spectra (for different methods, if different methods are specified)
       for Estimation_method in $list_of_methods; do
           echo "=================== Estimation method:   " $Estimation_method " ==================="
           echo "(4) Non-parametric multicomponent T2 estimation, please wait..."
           python run_real_data_script.py --path_to_folder='Recon_folder'/$subi/ --input='Data.nii.gz' --mask='mask.nii.gz' --minTE=10.68 --nTE=32 --TR=1000 --FA_method='spline' --FA_smooth='yes' --denoise='TV' --reg_method=$Estimation_method  --reg_matrix=$reg_matrix --savefig='yes' --savefig_slice=35 --numcores=-1 --myelin_T2=40
           # Optionally, you can disable the denosing by using --denoise='None', and the data can be denoised with any external software
           # The values: minTE, nTE, and TR, must be modified according to the parameters used in the scanner.
           
           # Total water content/proton-density correction for bias-field inhomogeneity
           echo "(5) Bias-field correction (using FAST-FSL) of Total water content map..."
           if [ "$Estimation_method" = "NNLS" ] || [ "$Estimation_method" = "T2SPARC" ]
           then
               recon_folder_name=recon_all_$Estimation_method
           else
               recon_folder_name=recon_all_$Estimation_method'-'$reg_matrix
           fi
           # Estimate bias-field from the proton density map
           fast -t 3 -n 3 -H 0.1 -I 4 -l 20.0 -b -o Recon_folder/$subi/$recon_folder_name/TWC Recon_folder/$subi/$recon_folder_name/TWC
           # Apply the correction to get the corrected map (i.e., corr = Raw/field-map)
           fslmaths Recon_folder/$subi/$recon_folder_name/TWC -div Recon_folder/$subi/$recon_folder_name/TWC_bias Recon_folder/$subi/$recon_folder_name/TWC
       done # Estimation_method
   else
       echo "Error: Data " $Path_to_Data " does not exist"
       echo $subi >> Recon_folder/subjects_with_problems.txt
   fi
done # subi
