# README for running LSTM and BERT in PYTORCH

## Prerequisit

Before running the models, please make sure that you have installed python3 and the packages below:
```
torch=1.7,horovod,psutil,pandas,transformers
```

## Run steps
### First, if you use AMD GPU, please set the environment variable to get rid of the incompatibility issue between MKL and libgomp in pytorch
```
export MKL_THREADING_LAYER=GNU
```

### Then, you could run the model

- For single gpu test:

    LSTM:
    ```
    python3 Pytorch_LSTMModel.py --batch_size=32 --warm_up=64 --num_test=2048 --distributed=False
    ````

    BERT:
    ```
    python3 Pytorch_BERTModel.py --batch_size=32 --warm_up=64 --num_test=2048 --distributed=False
    ```

- For single node test(8 gpus):
  
    LSTM:
    ```
    python3 -m torch.distributed.launch --nproc_per_node=8 Pytorch_LSTMModel.py  --batch_size=32 --warm_up=64 --num_test=2048 --distributed=True --dist_mode=native
    ```
    BERT:
    ```
    python3 -m torch.distributed.launch --nproc_per_node=8 Pytorch_BERTModel.py  --batch_size=32 --warm_up=64 --num_test=2048 --distributed=True --dist_mode=native
    ```

- Params:
  
    --warm_up: type=int, Num of warm up

    --num_test: type=int, Num of Test

    --batch_size: int, Num of batch size

    --distributed: type=bool, Whether to enable distributed pytorch training, all scenarios should enable it except single gpu

    --dist_mode: type=str, Distributed mode, such as horovod, or native

    --nproc_per_node: type=int, Num of gpu used in single node

- Extra setting
  
  If you want to set some extra variants, please add them before the `python` and run the command like this:

  ```
  ROCBLAS_LAYER=6 python3 Pytorch_BERTModel.py --batch_size=32 --warm_up=64 --num_test=2048 --distributed=False
  ```