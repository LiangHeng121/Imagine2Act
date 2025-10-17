exp=7task_imagine

tasks=(
     put_plate_in_colored_dish_rack
     put_toilet_roll_on_stand
     put_knife_in_knife_block
     phone_on_base
     stack_wine
     stack_cups
     place_cups
)
data_dir=./data/raw/test/
num_episodes=12
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="front"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
seed=1
save_frames=1  
checkpoint=./train_logs/7task_imagine/diffusion-C120-B16-lr1e-4-DI1-2-H3-DT100/best.pth
quaternion_format=xyzw

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    CUDA_LAUNCH_BLOCKING=1 python evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --diffusion_timesteps 100 \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --relative_action $relative_action \
    --num_history 3 \
    --test_model 3d_diffuser_actor \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "6D" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file ./eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions ./instructions/instructions.pkl \
    --variations 0 \
    --max_tries $max_tries \
    --max_steps 10 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04 \
    --quaternion_format $quaternion_format \
    --interpolation_length $interpolation_length \
    --dense_interpolation 1 \
    --headless 1 \
    --goal_frame_mode 3 \
    --loss_mode 7 \
    --save_frames $save_frames \
    --save_dir ./eval_logs/$exp/seed$seed/viz
done

