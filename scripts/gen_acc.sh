export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=48
# torchrun --nproc_per_node=4 --master_port=13456 tests/test_accept.py --target_group 0 1 2 3 --draft_group 0 1 2 3 --model JackFram/llama-68m --target meta-llama/Llama-2-70b-hf --T 0.6 --P 0.9 --M 256 --W 32 --dataset cnn --dst 68m-70b-acc.pt
# torchrun --nproc_per_node=4 --master_port=13456 tests/test_accept.py --target_group 0 1 2 3 --draft_group 0 1 2 3 --model princeton-nlp/Sheared-LLaMA-1.3B --target meta-llama/Llama-2-70b-hf --T 0.6 --P 0.9 --M 256 --W 32 --dataset cnn --dst 1.3b-70b-acc.pt
torchrun --standalone --nproc_per_node=8 --master_port=13456 tests/test_accept.py --target_group 0 1 2 3 4 5 6 7 --draft_group 0 1 2 3 4 5 6 7 --T 0.6 --P 0.9 --M 256 --W 31 --dataset cnn --dst new-7b-70b-acc.pt --compile
