#!/bin/bash
# ============================================================
# Imagine2Act - Image-to-3D Generation Pipeline
# This script performs multimodal scene imagination including:
#   1. Image generation with text prompts
#   2. Image resizing
#   3. Segmentation with Grounded-SAM
#   4. 3D reconstruction via TripoSR
#   5. Point cloud sampling
# Author: Imagine2Act Team
# ============================================================

# -------- Task Selection --------
TASK_NAME=$1      # Options: phone_on_base, place_cups, put_knife_in_knife_block, etc.
if [ -z "$TASK_NAME" ]; then
    echo "Usage: bash pipeline.sh <task_name>"
    echo "Example: bash pipeline.sh phone_on_base"
    exit 1
fi

# -------- Load Config --------
CONFIG_FILE="./configs/${TASK_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Loading config from $CONFIG_FILE"
eval $(python3 - <<END
import yaml, sys
cfg = yaml.safe_load(open("$CONFIG_FILE"))
for k, v in cfg.items():
    print(f'{k.upper()}="{v}"')
END
)

# -------- Print Loaded Variables --------
echo "Loaded config for task: $TASK_NAME"
echo "INPUT_IMAGE: $INPUT_IMAGE"
echo "PROMPT: $PROMPT"
echo "CLASSES_COMBINE: $CLASSES_COMBINE"
echo "CLASSES_REF: $CLASSES_REF"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "OUTPUT_REF_DIR: $OUTPUT_REF_DIR"

# Conda environments
ENV_GROUNDED="groundedsam"
ENV_TRIPOSR="triposr"

# -----------------------------
# Path Setup
# -----------------------------
mkdir -p "$OUTPUT_DIR" "$OUTPUT_REF_DIR"

RAW_OUTPUT="$OUTPUT_DIR/output_raw.png"
RESIZED_OUTPUT="$OUTPUT_DIR/output_resized.png"
SEGMENTED_OUTPUT="$OUTPUT_DIR/output_segmented.png"
FINAL_MESH="$OUTPUT_DIR/mesh.obj"
POINTCLOUD_OUTPUT="$OUTPUT_DIR/pcd.ply"

RESIZED_OUTPUT_REF="$OUTPUT_REF_DIR/output_resized.png"
SEGMENTED_OUTPUT_REF="$OUTPUT_REF_DIR/output_segmented.png"

# Save original image and prompt info
cp "$INPUT_IMAGE" "$OUTPUT_DIR/original.png"

# Step 1: Image Generation
echo "[1/7] Generating image..."
conda run -n "$ENV_GROUNDED" python3 utils/generate_image.py "$INPUT_IMAGE" "$PROMPT" "$RAW_OUTPUT"

# Step 2: Image Resize
echo "[2/7] Resizing image..."
conda run -n "$ENV_GROUNDED" python3 utils/resize.py "$RAW_OUTPUT" "$RESIZED_OUTPUT"

# Step 3: Segmentation (Grounded-SAM)
echo "[3/7] Running Grounded-SAM segmentation..."
pushd Grounded-Segment-Anything > /dev/null
conda run -n "$ENV_GROUNDED" python3 grounded_sam_imagine.py "../$RESIZED_OUTPUT" "../$SEGMENTED_OUTPUT" $CLASSES_COMBINE
popd > /dev/null

# Step 4: 3D Reconstruction (TripoSR)
echo "[4/7] Reconstructing 3D mesh with TripoSR..."
conda run -n "$ENV_TRIPOSR" python3 TripoSR/run_imagine.py "$SEGMENTED_OUTPUT" --output-dir "$OUTPUT_DIR"

# Step 5: Mesh to Point Cloud
echo "[5/7] Sampling point cloud..."
conda run -n "$ENV_TRIPOSR" python3 utils/mesh_to_pcd.py "$FINAL_MESH" "$POINTCLOUD_OUTPUT"

# Step 6: Original Image Resize
echo "[6/7] Resizing original image..."
conda run -n "$ENV_GROUNDED" python3 utils/resize.py "$OUTPUT_DIR/original.png" "$RESIZED_OUTPUT_REF"

# Step 7: Reference Object Segmentation + Reconstruction
echo "[7/7] Running reference segmentation and reconstruction..."
pushd Grounded-Segment-Anything > /dev/null
conda run -n "$ENV_GROUNDED" python3 grounded_sam_imagine.py "../$RESIZED_OUTPUT_REF" "../$SEGMENTED_OUTPUT_REF" $CLASSES_REF
popd > /dev/null

conda run -n "$ENV_TRIPOSR" python3 TripoSR/run_imagine.py "$SEGMENTED_OUTPUT_REF" --output-dir "$OUTPUT_REF_DIR"

echo "Pipeline completed successfully!"
