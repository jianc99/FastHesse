export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=8 --master_port=13456 tests/pipetree_benchmark.py --target_group 0 1 2 3 4 5 6 --draft_group 7 --T 0.6 --P 0.9 --M 256 --B 64 --growmap demo_tree.pt --Mode fast --compile
# torchrun --nproc_per_node=3 tests/pipetree_benchmark.py --target_group 0 1 --draft_group 2 --model princeton-nlp/Sheared-LLaMA-1.3B --target meta-llama/Llama-2-7b-hf --T 0.6 --P 0.9 --M 256 --B 16 --growmap demo_tree.pt