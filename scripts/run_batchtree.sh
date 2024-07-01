export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=48
# export ENABLE_INTRA_NODE_COMM=1
torchrun --standalone --nproc_per_node=4 tests/batchtree_benchmark.py --rank_group 0 1 2 3 --T 0.6 --P 0.9 --M 256 --B 2 --growmap demo_tree.pt --Mode fast