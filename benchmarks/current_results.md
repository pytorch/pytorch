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

Namespace(cnns=None, cuda_pointwise_block_count=None, cuda_pointwise_block_size=None, cuda_pointwise_loop_level=None, device='cuda', executor=None, fuser='te', group=['rnns'], hiddenSize=512, inputSize=512, miniBatch=64, nloops=100, numLayers=1, print_json=None, rnns=None, sep=' | ', seqLength=100, variable_lstms=False, warmup=10)

Benchmarking LSTMs...

|             name |          avg_fwd |          std_fwd |         info_fwd |          avg_bwd |          std_bwd |         info_bwd |
|            :---: |            :---: |            :---: |            :---: |            :---: |            :---: |  :---:           |
|            cudnn |            11.48 |          0.05617 |             None |            17.74 |           0.2205 |             None |
|             aten |             25.7 |           0.1008 |             None |            29.79 |            1.587 |             None |
|              jit |            16.21 |           0.5401 |             None |             53.5 |            9.683 |             None |
|       jit_premul |             14.4 |           0.3058 |             None |               52 |            8.515 |             None |
|  jit_premul_bias |            14.68 |          0.05764 |             None |            48.01 |             6.98 |             None |
|       jit_simple |            15.69 |           0.2142 |             None |            51.19 |            8.201 |             None |
|   jit_multilayer |            16.36 |           0.9769 |             None |            54.47 |            8.833 |             None |
|               py |             38.6 |            5.308 |             None |            51.96 |            7.255 |             None |

