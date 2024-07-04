export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=48
# export ENABLE_INTRA_NODE_COMM=1
torchrun --standalone --nproc_per_node=2 tests/seqspec_benchmark.py --rank_group 0 1 --T 0.6 --P 0.9 --M 256 --B 128 --growmap b32_batch.pt --Mode benchmark