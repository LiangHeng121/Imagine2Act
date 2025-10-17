main_dir=7task_imagine


export CUDA_VISIBLE_DEVICES=0

dataset=./data/packaged/train_imagine
valset=./data/packaged/val_imagine

lr=1e-4
dense_interpolation=1
interpolation_length=2
num_history=3
diffusion_timesteps=100
B=16
C=120
ngpus=1
quaternion_format=xyzw

CUDA_LAUNCH_BLOCKING=0 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory.py \
    --tasks stack_wine put_knife_in_knife_block phone_on_base put_plate_in_colored_dish_rack put_toilet_roll_on_stand place_cups stack_cups \
    --dataset $dataset \
    --valset $valset \
    --gripper_loc_bounds ./tasks/18_peract_tasks_location_bounds.json \
    --num_workers 5 \
    --train_iters 200000 \
    --embedding_dim $C \
    --use_instruction 1 \
    --instructions ./instructions/instructions.pkl \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq 2000 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 8 \
    --cache_size 600 \
    --cache_size_val 0 \
    --keypose_only 1 \
    --variations 0  \
    --lr $lr\
    --num_history $num_history \
    --cameras  front\
    --max_episodes_per_task -1 \
    --quaternion_format $quaternion_format \
    --run_log_dir diffusion-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps \
    # --checkpoint ~