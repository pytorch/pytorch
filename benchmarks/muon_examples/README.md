## Script to train a qwen model.

Training script adopted from https://github.com/MoonshotAI/Moonlight.

### Usage
```
OPTIMIZER_TYPE: one of "muon" or "pytorch_muon" or "adamw"

# DDP training
torchrun --nproc_per_node=8 benchmarks/muon_examples/train_ddp.py --model qwen --optimizer <OPTIMIZER_TYPE> --dataset openwebtext-100k --hidden_size 896 --lr 1e-3

# single GPU training
python3 benchmarks/muon_examples/train.py --model qwen --optimizer adamw --dataset openwebtext-100k --hidden_size 896 --lr 1e-3
```
