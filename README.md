# Hesse
## Installation
### Create Virtual Environment
``` bash
conda create -n hesse python=3.11
conda activate hesse
```

### Install Necessary Packages

``` bash
# pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install transformers==4.36.2
pip install protobuf
pip install sentencepiece
pip install datasets==2.16.1
pip install matplotlib
pip install wandb
pip install tiktoken
```

## Run Scripts
### Prepare Checkpoints
```bash
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
./scripts/prepare.sh $MODEL_REPO
```


## Performance on A100 80G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Sheared-LLaMA-2.7B  |  7.9 |   |   |  |
| Llama-2-7b  | 12.7  | 10.2  | 8.2  |   |
| Llama-2-13b  | 21.6 |   |   |   |
| Llama-2-70b | x  |   |   |   |
| vicuna-33b-v1.3 | 49.0  |   |   |   |

## Performance on A100 80G SXM
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-70b | x  | 59.0 | 37.5  | 27.7 |

## Performance on H100 80G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-7b  | 12.7  | 9.0  | 7.3  |   |

## Performance on 4090 24G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-7b  | 17.1  | 11.3  | 7.5  | 5.9  |
| Llama-2-70b | x  |  x | x  | 29.9  |
| vicuna-33b-v1.3 | x  | x  | 25.0  | x  |

## Performance on L40 48G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-7b  | 22.1  | 14.4  | 9.0  | 7.0  |
| Llama-2-70b | x  |  x | 69.9  | x  |

PP+TP Degree= 4 4 means the first and second pipeline stages are both doing tensor parallelism with degree=4.

| PP+TP Degree | 2 2 | 2 2 2 | 4 4 |
|---|---|---|---|
| Llama-2-7b  | 14.6  | 14.6 | 9.1 |