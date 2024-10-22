TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ OMP_NUM_THREADS=4 python -m torch.distributed.launch \
    --nproc_per_node 2 --master_port 1111 \
    scripts/train.py \
    --learning_rate 0.0001 \
    --tag "no word_feature (offical)" \
    --batch_size 7

