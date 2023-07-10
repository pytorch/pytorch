# Instructions for ```example_softmax_blockwise```

## Run ```example_softmax_blockwise```
```bash
# -D <xxx> : input 3-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes)
example_softmax_blockwise -D 4,128,2048 -v 1 1 1
```

Result
```
launch_and_time_kernel: grid_dim {64, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 0.0242877 ms, 259.039 GB/s, DeviceReduceSoftmax<256,M_C8_S1,K_C32_S8,InSrcVectorDim_1_InSrcVectorSize_8_OutDstVectorSize_8>
```
