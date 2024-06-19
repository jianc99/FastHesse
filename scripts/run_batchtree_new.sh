export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=48
torchrun --standalone --nproc_per_node=8 tests/batchtree_benchmark_new.py --target_group 0 1 2 3 4 5 6 7 --draft_group 0 1 2 3 4 5 6 7 --T 0.6 --P 0.9 --M 256 --B 2 --growmap demo_tree.pt --Mode benchmark --compile