#!/bin/bash

SPLIT=$1
shift
TASKS=("$@")   # e.g. bash data_generation.sh train phone_on_base place_cups stack_cups

if [ -z "$SPLIT" ] || [ ${#TASKS[@]} -eq 0 ]; then
    echo "Usage: bash data_generation.sh <split> <task1> [task2] [task3] ..."
    echo "Example: bash data_generation.sh train phone_on_base place_cups stack_cups"
    exit 1
fi

RAW_SAVE_PATH=./data/raw/${SPLIT}
PACKAGE_SAVE_PATH=./data/packaged/${SPLIT}

mkdir -p "$RAW_SAVE_PATH"
mkdir -p "$PACKAGE_SAVE_PATH"

echo "Generating RLBench data for split: $SPLIT"
echo "Tasks: ${TASKS[*]}"
echo "============================================="

for TASK in "${TASKS[@]}"; do
    echo "Processing task: $TASK"

    # Step 1: Generate raw RLBench episodes
    python ./RLBench/tools/dataset_generator.py \
        --save_path="$RAW_SAVE_PATH" \
        --tasks="$TASK" \
        --image_size=256,256 \
        --episodes_per_task=2 \
        --variations=1 \
        --all_variations=False

    # Step 2: Package into .dat format
    python data_preprocessing/package_rlbench.py \
        --data_dir="$RAW_SAVE_PATH" \
        --tasks="$TASK" \
        --output="$PACKAGE_SAVE_PATH" \
        --store_intermediate_actions=1

    echo "✅ Finished task: $TASK"
    echo "---------------------------------------------"
done

echo "All tasks completed! Data saved to $PACKAGE_SAVE_PATH"
