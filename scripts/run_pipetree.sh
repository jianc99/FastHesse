export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=6 --standalone --master_port=13456 tests/pipetree_benchmark.py --target_group 0 1 2 3 --draft_group 4 5 --T 0.6 --P 0.9 --M 256 --B 2 --growmap demo_tree.pt --Mode benchmark --compile
