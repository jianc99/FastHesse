export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=48

torchrun --standalone --nproc_per_node=8 --master_port=13456 tests/test_latency_new.py --maxlen 272 --declen_list 1 16 --prefixlen 128 --batch 2 --checkpoint_path checkpoints/meta-llama/Llama-2-70b-hf/model.pth --rank_group 0 1 2 3 4 5 6 7 --compile