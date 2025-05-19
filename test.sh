export MUJOCO_GL=osmesa

# Run 3D CAVLA evaluations on one of the LIBERO datasets
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "<path_to_checkpoint>" \
  --task_suite_name libero_spatial_cotdep \
  --use_depth True \
  --num_trials_per_task 50 

# Run OpenVLA-OFT evaluations on one of the LIBERO datasets
# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
#   --task_suite_name libero_spatial_cotdep \
#   --use_depth False \
#   --num_trials_per_task 50 