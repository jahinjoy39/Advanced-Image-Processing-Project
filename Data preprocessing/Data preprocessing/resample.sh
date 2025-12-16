#!/bin/bash

# --- Configuration ---
# Set the base directory where your subject folders are located
BASE_INPUT_DIR="/data/users3/gnagaboina1/AIP/Data/NIfTI_Converted_Series"

# Set the path for the common mask file (used as the resampling master)
COMMON_MASK="/data/users3/gnagaboina1/mask_common.nii"

# Set the output directory where the final resampled files will be saved
OUTPUT_DIR="/data/users3/gnagaboina1/AIP/Data/Downsampled_Output" 

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting NIfTI Processing Script..."
echo "Input Directory: $BASE_INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Master Mask: $COMMON_MASK"
echo "--------------------------------------------------"


# --- Processing Loop ---
for subject_folder in "$BASE_INPUT_DIR"/*/; do
    
    if [ -d "$subject_folder" ]; then
        
        subject_id=$(basename "$subject_folder")
        input_file=$(find "$subject_folder" -maxdepth 1 -type f -name "*.nii" | head -n 1)

        if [ -n "$input_file" ]; then
            
            echo ""
            echo "--> Processing Subject: ${subject_id}"
            echo "    Input File: $(basename "$input_file")"

            # Define temporary file names
            temp_deoblique_file="${OUTPUT_DIR}/${subject_id}_deoblique_temp.nii"
            
            # 1. DE-OBLIQUE STEP (FIXED SYNTAX HERE)
            echo "    1. De-obliquing..."
            # The input file is specified as the last argument, NOT using -input
            3dWarp -deoblique -prefix "$temp_deoblique_file" "$input_file"
            
            # Check the exit status of 3dWarp to ensure it succeeded
            if [ $? -ne 0 ]; then
                echo "    !!! ERROR: 3dWarp failed for ${subject_id}. Skipping to next subject."
                continue 
            fi
            
            # 2. RESAMPLE STEP
            echo "    2. Resampling (3dresample) to master mask geometry..."
            3dresample \
                -master "$COMMON_MASK" \
                -prefix "${OUTPUT_DIR}/${subject_id}_resampled_afni" \
                -input "$temp_deoblique_file" \
                -rmode Cu
            
            # 3. CONVERT and CLEANUP
            echo "    3. Converting to NIfTI and cleaning up temporary files..."
            
            # Convert the resulting AFNI dataset back to a NIfTI file (.nii)
            3dAFNItoNIFTI "${OUTPUT_DIR}/${subject_id}_resampled_afni+tlrc.BRIK"
            
            # Remove the temporary AFNI BRIK/HEAD files
            rm "${OUTPUT_DIR}/${subject_id}_resampled_afni+tlrc.BRIK"
            rm "${OUTPUT_DIR}/${subject_id}_resampled_afni+tlrc.HEAD"
            
            # Remove the temporary de-obliqued NIfTI file
            rm "$temp_deoblique_file" 
            
            # Rename the final NIfTI file to be cleaner
            mv "${OUTPUT_DIR}/${subject_id}_resampled_afni.nii" "${OUTPUT_DIR}/${subject_id}_resampled.nii"

            echo "    -> SUCCESS: Output saved as ${subject_id}_resampled.nii"
        else
            echo "Warning: No .nii file found in folder: $subject_folder"
        fi
    fi
done

echo ""
echo "--------------------------------------------------"
echo "--- Script finished. Check ${OUTPUT_DIR} for results. ---"