#!/usr/bin/env bash
# Run LoRA text2img + ControlNet generation for all 10 artists across 4 L4 GPUs.
#
# Strategy: shard 10 artists across 4 GPUs (3,3,2,2). For each pipeline (A then B),
# launch 4 background processes pinned via CUDA_VISIBLE_DEVICES, wait for all,
# then proceed to the next pipeline.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/shared/courseSharedFolders/163602outer/163602/cs1090b-gpu/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Artists per GPU. Keep all-lowercase slugs that match lora_ckpts/<slug>/ folders.
# GPU 2 skipped — another process occupies ~12GB there.
GPU0_ARTISTS=("andrei_rublev" "claude_monet" "eugene_delacroix" "gustave_courbet")
GPU1_ARTISTS=("kazimir_malevich" "mikhail_vrubel" "paul_cezanne")
GPU3_ARTISTS=("titian" "vasiliy_kandinskiy" "vincent_van_gogh")
ACTIVE_GPUS=(0 1 3)

run_pipeline() {
    local pipeline="$1"; shift
    local script="$1"; shift
    local extra_args=("$@")

    echo "==================================================================="
    echo "Pipeline: ${pipeline}"
    echo "Script:   ${script}"
    echo "==================================================================="

    local pids=()
    local pid_gpus=()

    for gpu in "${ACTIVE_GPUS[@]}"; do
        local arr_name="GPU${gpu}_ARTISTS[@]"
        local artists=("${!arr_name}")
        local log="${LOG_DIR}/${pipeline}_gpu${gpu}.log"
        echo "[GPU ${gpu}] artists: ${artists[*]} -> ${log}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${script}" \
            --artists "${artists[@]}" \
            "${extra_args[@]}" \
            > "${log}" 2>&1 &
        pids+=($!)
        pid_gpus+=("${gpu}")
    done

    local rc=0
    local i=0
    for pid in "${pids[@]}"; do
        local gpu="${pid_gpus[$i]}"
        if wait "${pid}"; then
            echo "[GPU ${gpu}] ${pipeline} OK (pid=${pid})"
        else
            local code=$?
            echo "[GPU ${gpu}] ${pipeline} FAILED (pid=${pid}, exit=${code})"
            rc=1
        fi
        i=$((i+1))
    done

    return ${rc}
}

cd "${PROJECT_DIR}"

echo "============= PIPELINE A: LoRA Text2Img ============="
run_pipeline "t2i" "${PROJECT_DIR}/Phase1_Augmentation_LoRA_Text2Img.py"
A_RC=$?
echo "Pipeline A exit: ${A_RC}"

echo "============= PIPELINE B: LoRA + ControlNet ============="
run_pipeline "cn" "${PROJECT_DIR}/Phase1_Augmentation_LoRA_ControlNet.py" --save-canny
B_RC=$?
echo "Pipeline B exit: ${B_RC}"

echo "==================================================================="
echo "DONE.  text2img=${A_RC}  controlnet=${B_RC}"
echo "Outputs:"
echo "  ${PROJECT_DIR}/data/generated_lora/"
echo "  ${PROJECT_DIR}/data/generated_lora_canny/"
echo "Logs:    ${LOG_DIR}/"
exit $((A_RC + B_RC))
