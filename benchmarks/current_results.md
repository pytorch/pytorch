# Old Fuser + Legacy Executor

```
$ python -m fastrnns.bench --fuser=old --group=rnns --sep " | "
```

Namespace(cnns=None, cuda_pointwise_block_count=None, cuda_pointwise_block_size=None, cuda_pointwise_loop_level=None, device='cuda', executor=None, fuser='old', group=['rnns'], hiddenSize=512, inputSize=512, miniBatch=64, nloops=100, numLayers=1, print_json=None, rnns=None, sep=' | ', seqLength=100, variable_lstms=False, warmup=10)

Benchmarking LSTMs...

|            name |          avg_fwd |          std_fwd |         info_fwd |          avg_bwd |          std_bwd |         info_bwd |
|           :---: |            :---: |            :---: |            :---: |            :---: |            :---: |  :---:           |
|           cudnn |            11.37 |          0.06225 |             None |             17.8 |           0.2296 |             None |
|            aten |            25.76 |          0.06663 |             None |            29.74 |           0.8339 |             None |
|             jit |            17.27 |           0.7483 |             None |            27.41 |            1.264 |             None |
|      jit_premul |            14.97 |          0.05655 |             None |            22.17 |            2.878 |             None |
| jit_premul_bias |            15.15 |           0.1199 |             None |            21.78 |            2.648 |             None |
|      jit_simple |             16.9 |           0.2016 |             None |            26.88 |            1.362 |             None |
|  jit_multilayer |            16.97 |           0.1585 |             None |            27.26 |            1.387 |             None |
|              py |            37.87 |            5.554 |             None |            51.88 |            7.287 |             None |

# TE Fuser + Profiling Executor

```
$ python -m fastrnns.bench --fuser=te --group=rnns --sep " | "
```

Namespace(cnns=None, cuda_pointwise_block_count=None, cuda_pointwise_block_size=None, cuda_pointwise_loop_level=None, device='cuda', executor=None, fuser='te', group=['rnns'], hiddenSize=512, inputSize=512, miniBatch=64, nloops=100, numLayers=1, print_json=None, rnns=None, sep=' ', seqLength=100, variable_lstms=False, warmup=10)

Benchmarking LSTMs...

|            name |         avg_fwd  |        std_fwd   |      info_fwd    |      avg_bwd     |     std_bwd      |   info_bwd |
|           :---: |            :---: |            :---: |            :---: |            :---: |            :---: |  :---:     |
|           cudnn |           11.58  |        0.05633   |          None    |         17.9     |      0.2226      |       None |
|            aten |           25.77  |         0.1117   |          None    |        30.46     |       2.006      |       None |
|             jit |           17.11  |          1.067   |          None    |        54.39     |       9.064      |       None |
|      jit_premul |           15.32  |         0.9256   |          None    |        53.49     |       7.958      |       None |
| jit_premul_bias |            15.2  |         0.1849   |          None    |        48.74     |       6.442      |       None |
|      jit_simple |           16.43  |         0.3517   |          None    |        53.16     |       7.327      |       None |
|  jit_multilayer |           17.27  |          1.018   |          None    |        56.31     |       8.865      |       None |
|              py |           41.06  |          7.026   |          None    |        56.39     |       10.65      |       None |
