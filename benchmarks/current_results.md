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
|            cudnn |            11.57 |          0.07775 |             None |            18.04 |           0.4799 |             None |
|             aten |            25.75 |          0.06975 |             None |            29.98 |            1.738 |             None |
|              jit |             16.3 |           0.9028 |             None |            30.58 |            4.496 |             None |
|       jit_premul |            14.26 |           0.1378 |             None |            27.86 |            3.647 |             None |
|  jit_premul_bias |            14.59 |           0.1666 |             None |            27.16 |            3.558 |             None |
|       jit_simple |            15.77 |           0.5976 |             None |            29.86 |            4.391 |             None |
|   jit_multilayer |            16.26 |              0.8 |             None |            30.04 |            3.947 |             None |
|               py |            41.18 |            7.343 |             None |            56.05 |            9.657 |             None |
