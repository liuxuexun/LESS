TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ OMP_NUM_THREADS=4 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 1112 \
    scripts/train.py \
    --learning_rate 0.0001 \
    --tag "no word_feature (offical)" \
    --batch_size 7
    # --use_checkpoint /data/liuxuexun/code/NIPS/LESS/outputs/2024-10-19_14-25-22/checkpoint_last.pth

