export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
export OMP_NUM_THREADS=48

torchrun --standalone --nproc_per_node=8 --master_port=13456 tests/test_latency.py --maxlen 272 --declen_list 1 2 3 --prefixlen 128 --batch 1 --checkpoint_path checkpoints/meta-llama/Llama-2-70b-hf/model.pth --rank_group 0 1 2 3 4 5 6 7 --compile