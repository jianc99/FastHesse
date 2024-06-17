export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=48

torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency_new.py --maxlen 288 --declen_list 1 2 4 8 16 32 --prefixlen 128 --batch 1 --compile --checkpoint_path /home/jianc2/gpt-fast/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --rank_group 0 1 2 3 