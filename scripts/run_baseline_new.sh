export CUDA_VISIBLE_DEVICES=0,1,3
export OMP_NUM_THREADS=48
export ENABLE_INTRA_NODE_COMM=1
torchrun --standalone --nproc_per_node=3 --master_port=13456 tests/baseline_new.py --B 1 --checkpoint_path /checkpoints/meta-llama/Llama-2-7b-hf/model.pth --compile --rank_group 0 1 2