
# Run 3D-CAVLA default fine-tuning on one of the LIBERO datasets

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "<path_to_modified_libero_rlds_cotdep>" \
  --dataset_name libero_spatial_cotdep \
  --run_root_dir ./runs \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_depth True \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 50000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "<wandb_entity>" \
  --wandb_project "3dcavla-experiments" \
  --run_id_note libero-spatial-cotdep-3dcavla

# Run baseline openvla-oft fine-tuning on one of the LIBERO datasets

#   torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_custom.py \
#   --vla_path openvla/openvla-7b \
#   --data_root_dir "<path_to_modified_libero_rlds_cotdep>" \
#   --dataset_name libero_spatial_cotdep \
#   --run_root_dir ./runs \
#   --use_l1_regression True \
#   --use_diffusion False \
#   --use_film False \
#   --num_images_in_input 2 \
#   --use_proprio True \
#   --use_depth False \
#   --batch_size 8 \
#   --learning_rate 5e-4 \
#   --num_steps_before_decay 50000 \
#   --max_steps 150005 \
#   --save_freq 50000 \
#   --save_latest_checkpoint_only False \
#   --image_aug True \
#   --lora_rank 32 \
#   --wandb_entity "<wandb_entity>" \
#   --wandb_project "3dcavla-experiments" \
#   --run_id_note libero-spatial-cotdep-ovlaoft