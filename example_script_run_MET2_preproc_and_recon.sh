
# Define the path to your data
Data=Multi_echo_T2.nii.gz
Path='/media/Disco1T/MET2_Data'

# Create the reconstruction folder
mkdir Recon_folder

# Uncomment one line from the list to select one method:
#Estimation_method='X2-I'
#Estimation_method='X2-L1'
Estimation_method='X2-L2'
#Estimation_method='L_curve-I'
#Estimation_method='L_curve-L1'
#Estimation_method='L_curve-L2'
#Estimation_method='GCV-I'
#Estimation_method='GCV-L1'
#Estimation_method='GCV-L2'
#Estimation_method='NNLS'

list_of_subjects="
SUBJECT_1
SUBJECT_2
"

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

       mkdir Recon_folder/$subi

       # Copy data to local folder
       echo "(1) Copy data to local folder"
       fslmaths $Path_to_Data  Recon_folder/$subi/Data.nii.gz

       # Brain extraction (BET) using FSL
       echo "(2) BET, please wait..."
       fslmaths Recon_folder/$subi/Data.nii.gz -Tmean Recon_folder/$subi/Data_avg.nii.gz
       bet Recon_folder/$subi/Data_avg.nii.gz Recon_folder/$subi/Data_mask -m -v -f 0.4
       mv Recon_folder/$subi/Data_mask_mask.nii.gz Recon_folder/$subi/mask.nii.gz

       ### -------------------------------------------------------------------------------- ###
       ### Two optional steps to reduce the noise and gibbs ringing artifacts using MRtrix3 ###
       ### -------------------------------------------------------------------------------- ###
       # Signal denoising using Mrtrix3: patch-level PCA
       echo "(3) Denoising -  using Mrtrix3"
       dwidenoise Recon_folder/$subi/Data.nii.gz  Recon_folder/$subi/Data.nii.gz -noise Recon_folder/$subi/noise_map.nii.gz -mask Recon_folder/$subi/mask.nii.gz -force

       # Remove Gibbs Ringing Artifacts using MRtrix3
       echo "(4) Remove Gibbs Ringing Artifacts, please wait..."
       mrdegibbs Recon_folder/$subi/Data.nii.gz Recon_folder/$subi/Data.nii.gz -force
       ### -------------------------------------------------------------------------------- ###
   
       # Estimate T2 spectra
       echo "(5) Non-parametric multicomponent T2 estimation, please wait..."
       python run_real_data_script.py --minTE=10.00 --nTE=32 --TR=1000  --FA_method='spline' --FA_smooth='yes' --denoise='None' --reg_method=$Estimation_method --path_to_folder='Recon_folder'/$subi/ --input='Data.nii.gz'  --mask='mask.nii.gz' --savefig='yes' --savefig_slice=40 --numcores=-1 --myelin_T2=40

       # Total water content/proton-density correction for bias-field inhomogeneity
       echo "(6) Bias-field correction (using FAST-FSL) of proton density map, please wait..."
       # Estimate bias-field from the proton density map
       fast -t 3 -n 3 -H 0.1 -I 4 -l 20.0 -b -o Recon_folder/$subi/recon_all_$Estimation_method/Ktotal Recon_folder/$subi/recon_all_$Estimation_method/Ktotal
       # Apply the correction to get the corrected map (i.e., corr = Raw/field-map)
       fslmaths Recon_folder/$subi/recon_all_$Estimation_method/Ktotal -div Recon_folder/$subi/recon_all_$Estimation_method/Ktotal_bias Recon_folder/$subi/recon_all_$Estimation_method/Ktotal
       # ----------------------
       # Save images (FSL tool)
       slices Recon_folder/$subi/recon_all_$Estimation_method/fM      -o  Recon_folder/$subi/recon_all_$Estimation_method/fM.gif
       slices Recon_folder/$subi/recon_all_$Estimation_method/Ktotal  -o  Recon_folder/$subi/recon_all_$Estimation_method/Ktotal.gif
       slices Recon_folder/$subi/recon_all_$Estimation_method/T2_IE   -o  Recon_folder/$subi/recon_all_$Estimation_method/T2_IE.gif
       slices Recon_folder/$subi/recon_all_$Estimation_method/fCSF    -o  Recon_folder/$subi/recon_all_$Estimation_method/fCSF.gif
   else
       echo "Error: Data " $Path_to_Data " does not exist"
       echo $subi >> Recon_folder/subjects_with_problems.txt
   fi
done

