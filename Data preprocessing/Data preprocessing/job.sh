#!/bin/bash
#SBATCH --job-name=dcm2niix_array
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=25G
#SBATCH --partition=qTRD

# ---- Paths ----
INPUT_ROOT_DIR="/data/users3/gnagaboina1/AIP/Data/MRI-T1_Series"
OUTPUT_ROOT_DIR="/data/users3/gnagaboina1/AIP/Data/NIfTI_Converted_Series"
INPUT_LIST="/data/users3/gnagaboina1/AIP/Data/code/input.txt"

# ---- Create logs directory ----
LOG_DIR="${OUTPUT_ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Redirect STDOUT and STDERR to logs
exec > >(tee "${LOG_DIR}/dcm2niix_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out")
exec 2> >(tee "${LOG_DIR}/dcm2niix_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)

echo "Starting task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

# ---- Load dcm2niix only ----
module load dcm2niix/1.0.20220720

# ---- Extract directory name for this array task ----
TARGET_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${INPUT_LIST}" | tr -d '\r\n')
TARGET_DIR=${TARGET_DIR%/}  # remove trailing /

# ---- Build full paths ----
FULL_INPUT_DIR="${INPUT_ROOT_DIR}/${TARGET_DIR}"
FULL_OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${TARGET_DIR}"

# ---- Create output directory ----
mkdir -p "${FULL_OUTPUT_DIR}"

echo "Input directory:  ${FULL_INPUT_DIR}"
echo "Output directory: ${FULL_OUTPUT_DIR}"

# ---- Run dcm2niix ----
dcm2niix -o "${FULL_OUTPUT_DIR}" -f "%p_T1w" -z n "${FULL_INPUT_DIR}"

if [ $? -eq 0 ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} completed successfully."
else
    echo "Task ${SLURM_ARRAY_TASK_ID} FAILED." >&2
fi
