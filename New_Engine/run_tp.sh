export CUDA_VISIBLE_DEVICES=3,4,5,6
export OMP_NUM_THREADS=48

torchrun --nproc_per_node=2 --master_port=13456 tp_benchmark.py --maxlen 288 --declen 1 --prefixlen 128 --batch 128 --compile --checkpoint_path ~/gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth 