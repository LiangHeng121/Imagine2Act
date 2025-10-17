#!/bin/bash
# =====================================================================
# Imagine2Act - Scene Alignment and Fusion Pipeline
#
# This script automates the full 3D alignment process for multiple episodes:
#   1. Depth preprocessing
#   2. RGB and camera intrinsic preparation
#   3. Segmentation with Grounded-SAM
#   4. Pose estimation with FoundationPose
#   5. Point cloud fusion and rendering
#   6. Projection and world coordinate transformation
#
# Author: Imagine2Act Team
# =====================================================================

# -------- Task Selection --------
TASK_NAME=$1      # Options: phone_on_base, place_cups, put_knife_in_knife_block, etc.
if [ -z "$TASK_NAME" ]; then
    echo "Usage: bash pipeline.sh <task_name> <split>"
    echo "Example: bash pipeline.sh phone_on_base train"
    exit 1
fi

SPLIT=$2  # Options: train, val, test
if [ -z "$SPLIT" ]; then
    echo "Usage: bash pipeline.sh <task_name> <split>"
    echo "Example: bash pipeline.sh phone_on_base train"
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

ROOT_EP_DIR="./imagine2act/data/raw/${SPLIT}/${TASK_NAME}/variation0/episodes"

# -------- Print Loaded Variables --------
echo "Loaded config for task: $TASK_NAME"
echo "ROOT_EP_DIR: $ROOT_EP_DIR"
echo "IMAGINE_PCD: $IMAGINE_PCD"
echo "ORIGIN_MESH: $ORIGIN_MESH"
echo "CLASSES_REF: $CLASSES_REF"
echo "CLASSES_MANIP: $CLASSES_MANIP"
echo "MESH_SCALE: $MESH_SCALE"
echo "SCALE_2: $SCALE_2"

# -----------------------------
# Conda Environment Settings
# -----------------------------
ENV_GROUNDED="groundedsam"
ENV_FOUNDATIONPOSE="foundationpose"
ENV_TRIPOSR="triposr"

# Specify episodes to process (e.g., INCLUDE_EPISODES=(6 8 12))
INCLUDE_EPISODES=(0)

# -----------------------------
# Episode Loop
# -----------------------------
for EP_DIR in "$ROOT_EP_DIR"/episode*/; do
    EP_NUM=$(basename "$EP_DIR" | sed 's/episode//')

    # # Skip episodes not listed
    # if [[ ! " ${INCLUDE_EPISODES[@]} " =~ " ${EP_NUM} " ]]; then
    #     continue
    # fi

    echo "===================================================="
    echo "Processing episode $EP_NUM ..."
    echo "===================================================="

    IMAGINE_DIR="$EP_DIR/imagine"
    OUTPUT_DIR="$IMAGINE_DIR"
    mkdir -p "$OUTPUT_DIR"

    DEPTH_IMAGE="${EP_DIR}front_depth/0.png"
    RGB_IMAGE="${EP_DIR}front_rgb/0.png"

    DEMO_DATA_DIR="$OUTPUT_DIR/demo_data"
    mkdir -p "$DEMO_DATA_DIR/depth" "$DEMO_DATA_DIR/rgb" "$DEMO_DATA_DIR/masks" "$DEMO_DATA_DIR/mesh"


    # Step 1: Convert Depth to 16-bit PNG
    echo "[1/16] Converting depth to 16-bit..."
    conda run -n "$ENV_GROUNDED" python utils/depth_to_uint16.py "$DEPTH_IMAGE" "$DEMO_DATA_DIR/depth/image.png" || { echo "Depth conversion failed."; continue; }


    # Step 2: Copy RGB Image
    echo "[2/16] Copying RGB image..."
    cp "$RGB_IMAGE" "$DEMO_DATA_DIR/rgb/image.png"


    # Step 3: Write Camera Intrinsics
    echo "[3/16] Writing camera intrinsics..."
    cat << EOF > "$DEMO_DATA_DIR/cam_K.txt"
351.6771208  0  128
0 351.6771208  128
0  0  1
EOF


    # Step 4: Generate Binary Mask
    echo "[4/16] Generating black-white mask..."
    pushd Grounded-Segment-Anything > /dev/null
    conda run -n "$ENV_GROUNDED" python3 grounded_sam_imagine_black_white.py "$RGB_IMAGE" "$DEMO_DATA_DIR/masks/image.png" $CLASSES_REF || { popd; echo "Mask generation failed."; continue; }
    popd > /dev/null


    # Step 5: Segmentation (CLASSES_REF)
    echo "[5/16] Running semantic segmentation (CLASSES_REF)..."
    pushd Grounded-Segment-Anything > /dev/null
    conda run -n "$ENV_GROUNDED" python3 grounded_sam_imagine.py "$RGB_IMAGE" "$OUTPUT_DIR/rgb_segmented_1024.png" $CLASSES_REF || { popd; echo "Segmentation failed."; continue; }
    popd > /dev/null


    # Step 6: Segmentation (CLASSES_MANIP)
    echo "[6/16] Running semantic segmentation (CLASSES_MANIP)..."
    pushd Grounded-Segment-Anything > /dev/null
    conda run -n "$ENV_GROUNDED" python3 grounded_sam_imagine.py "$RGB_IMAGE" "$OUTPUT_DIR/rgb_segmented_1024_2.png" $CLASSES_MANIP || { popd; echo "Segmentation failed."; continue; }
    popd > /dev/null


    # Step 7: Copy Mesh for FoundationPose
    echo "[7/16] Copying mesh to FoundationPose directory..."
    cp "$ORIGIN_MESH" "$DEMO_DATA_DIR/mesh/textured_simple.obj"


    # Step 8: Run FoundationPose
    echo "[8/16] Running FoundationPose..."
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    pushd FoundationPose > /dev/null
    conda run -n "$ENV_FOUNDATIONPOSE" xvfb-run --auto-servernum python3 run_demo.py \
        --mesh_file "$DEMO_DATA_DIR/mesh/textured_simple.obj" \
        --test_scene_dir "$DEMO_DATA_DIR" \
        --est_refine_iter 5 --track_refine_iter 2 \
        --debug 3 --debug_dir "$OUTPUT_DIR/fp_output" \
        --mesh_scale "$MESH_SCALE" || { popd; echo "FoundationPose failed."; continue; }
    popd > /dev/null


    # Step 9: Copy and Adjust Pose
    echo "[9/16] Copying and refining pose..."
    cp "$OUTPUT_DIR/fp_output/ob_in_cam/image.txt" "$OUTPUT_DIR/pose.txt"
    conda run -n "$ENV_FOUNDATIONPOSE" python utils/flip_pose.py "$OUTPUT_DIR/pose.txt" "$OUTPUT_DIR/pose_flip.txt"

    # conda run -n "$ENV_TRIPOSR" python bg/scale_pose_t.py "$OUTPUT_DIR/pose.txt" "$OUTPUT_DIR/pose_scaled.txt" --scale 4.5
    # conda run -n "$ENV_FOUNDATIONPOSE" python bg/flip_pose.py "$OUTPUT_DIR/pose_scaled.txt" "$OUTPUT_DIR/pose_flip.txt"


    # Step 10: Align and Fuse TripoSR Point Cloud with Scene
    echo "[10/16] Aligning and fusing point clouds..."
    conda run -n "$ENV_TRIPOSR" python utils/align_with_fp.py \
        "$RGB_IMAGE" "$DEPTH_IMAGE" \
        "$OUTPUT_DIR/rgb_segmented_1024.png" "$OUTPUT_DIR/rgb_segmented_1024_2.png" \
        "$IMAGINE_PCD" "$OUTPUT_DIR/pose_flip.txt" \
        "$OUTPUT_DIR/fused_scene.ply" "$SCALE_2" || { echo "Fusion failed."; continue; }


    # Step 11: Render Fused Scene
    echo "[11/16] Rendering fused point cloud..."
    conda run -n "$ENV_TRIPOSR" python utils/render_ply.py \
        "$OUTPUT_DIR/fused_scene_filtered.ply" \
        "$OUTPUT_DIR/fused_scene_rendered.png" || { echo "Render failed."; continue; }


    # Step 12: Write Camera Extrinsics
    echo "[12/16] Writing camera extrinsics..."
    cat << EOF > "$OUTPUT_DIR/extrinsic.txt"
1.19209290e-07 -4.22617942e-01 -9.06307936e-01  1.34999919e+00
-1.00000000e+00 -5.96046448e-07  1.49011612e-07  3.71546562e-08
-5.66244125e-07  9.06307936e-01 -4.22617912e-01  1.57999933e+00
0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00
EOF


    # Step 13: Project to Dense Point Image
    echo "[13/16] Projecting fused scene to dense point images..."
    conda run -n "$ENV_TRIPOSR" python utils/project_ply_to_dense_image.py "$OUTPUT_DIR/fused_scene_filtered.ply" "$DEPTH_IMAGE" "$OUTPUT_DIR/fused_scene_filtered_dense_point_image.npy"
    conda run -n "$ENV_TRIPOSR" python utils/project_ply_to_dense_image.py "$OUTPUT_DIR/fused_scene_full.ply" "$DEPTH_IMAGE" "$OUTPUT_DIR/fused_scene_full_dense_point_image.npy"


    # Step 14: Convert to World Coordinates
    echo "[14/16] Converting dense point cloud to world coordinates..."
    conda run -n "$ENV_TRIPOSR" python utils/convert_dense_pcd_to_world.py \
        "$OUTPUT_DIR/fused_scene_filtered_dense_point_image.npy" \
        "$OUTPUT_DIR/extrinsic.txt" \
        "$OUTPUT_DIR/fused_scene_filtered_dense_point_image_world.npy"


    # Step 15: Secondary Segmentation on Rendered Scene
    echo "[15/16] Segmenting rendered fused scene..."
    pushd Grounded-Segment-Anything > /dev/null
    conda run -n "$ENV_GROUNDED" python3 grounded_sam_imagine.py "$OUTPUT_DIR/fused_scene_rendered.png" "$OUTPUT_DIR/rgb_segmented_1024_3.png" $CLASSES_MANIP || { popd; echo "Segmentation failed."; continue; }
    popd > /dev/null


    # Step 16: Compute Rigid Transformation Between Masks
    echo "[16/16] Computing rigid transform from semantic masks..."
    conda run -n "$ENV_TRIPOSR" python utils/compute_rigid_transform_from_masks.py \
        "$OUTPUT_DIR/rgb_segmented_1024_2.png" \
        "$OUTPUT_DIR/rgb_segmented_1024_3.png" \
        "$OUTPUT_DIR/fused_scene_full_dense_point_image.npy" \
        "$OUTPUT_DIR/extrinsic.txt" \
        "$OUTPUT_DIR/transform.txt" || { echo "Rigid transform computation failed."; continue; }

    echo "✅ Episode $EP_NUM completed successfully!"
    echo "----------------------------------------------------"
done

echo "✅ All episodes processed successfully."
