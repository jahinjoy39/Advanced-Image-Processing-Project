#!/bin/bash
#SBATCH --job-name=aip_preproc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --partition=qTRD
#SBATCH --array=1-1000%20
#SBATCH --output=/data/users3/gnagaboina1/AIP/Data/processed_data_test/logs/preproc_%A_%a.out
#SBATCH --error=/data/users3/gnagaboina1/AIP/Data/processed_data_test/logs/preproc_%A_%a.err

# ===============================================================
# Modules & Paths
# ===============================================================
module load fsl/6.0.7.17
module load ants/2.4.2

DATA_DIR="/data/users3/gnagaboina1/AIP/Data/NIfTI_Converted_Series_test"
LIST_FILE="/data/users3/gnagaboina1/AIP/Data/code/test_mni.txt"    # ← absolute path!
OUTPUT_DIR="/data/users3/gnagaboina1/AIP/Data/processed_data_test"
MNI_TEMPLATE="${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz"

mkdir -p ${OUTPUT_DIR}/{01_raw,02_bet,03_n4,04_mni_rigid,logs}

# ===============================================================
# Read subject base name (no extension) from process.txt
# ===============================================================
BASE_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE" | tr -d '[:space:]')
[[ -z "$BASE_ID" ]] && echo "No entry at line ${SLURM_ARRAY_TASK_ID} → exiting" && exit 0

echo "=== Task ${SLURM_ARRAY_TASK_ID}: Processing ${BASE_ID} ==="

# Find the actual file (supports both .nii and .nii.gz)
INPUT_NII=$(find "$DATA_DIR" -type f \( -name "${BASE_ID}.nii" -o -name "${BASE_ID}.nii.gz" \) | head -n 1)

if [[ -z "$INPUT_NII" ]]; then
    echo "ERROR: No NIfTI file found for ${BASE_ID} in ${DATA_DIR}"
    exit 1
fi

echo "Found: $(basename "$INPUT_NII")"

# Output files (use BASE_ID as folder/filename base)
raw="${OUTPUT_DIR}/01_raw/${BASE_ID}.nii.gz"
bet="${OUTPUT_DIR}/02_bet/${BASE_ID}_bet.nii.gz"
n4="${OUTPUT_DIR}/03_n4/${BASE_ID}_bet_n4.nii.gz"
warped="${OUTPUT_DIR}/04_mni_rigid/${BASE_ID}_to_MNI.nii.gz"
affine_mat="${OUTPUT_DIR}/04_mni_rigid/${BASE_ID}_0GenericAffine.mat"

# 1) Copy raw
cp "$INPUT_NII" "$raw"

# 2) BET
bet "$raw" "$bet" -R -f 0.5 -g 0

# 3) N4 — bulletproof (no non-positive warning)
tmp_pos="${OUTPUT_DIR}/03_n4/${BASE_ID}_tmp_pos.nii.gz"
tmp_n4="${OUTPUT_DIR}/03_n4/${BASE_ID}_tmp_n4.nii.gz"

fslmaths "$bet" -mul 10000 -add 1 "$tmp_pos"
N4BiasFieldCorrection -d 3 -i "$tmp_pos" -o "$tmp_n4" -b [200] -c [50x50x30x20,1e-6] -s 3 -v 1
fslmaths "$tmp_n4" -div 10000 "$n4"
rm -f "$tmp_pos" "$tmp_n4"

# 4) Rigid registration to MNI152 1mm
antsRegistration -d 3 --verbose 1 \
    -r [${MNI_TEMPLATE},${n4},1] \
    -m MI[${MNI_TEMPLATE},${n4},1,32] \
    -t Rigid[0.1] \
    -c [1000x500x250,1e-6,10] \
    -s 3x2x1mm \
    -f 4x2x1 \
    -o [${OUTPUT_DIR}/04_mni_rigid/${BASE_ID}_,${warped}]

# 5) Apply transform
antsApplyTransforms -d 3 -i "$n4" -r ${MNI_TEMPLATE} -o "$warped" -t "$affine_mat" -n Linear

echo "=== SUCCESS: ${BASE_ID} finished ==="
