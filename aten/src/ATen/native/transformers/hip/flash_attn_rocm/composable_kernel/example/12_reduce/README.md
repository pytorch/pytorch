# Instructions for ```example_reduce_blockwise```

## Run ```example_reduce_blockwise```
```bash
# -D <xxx> : input 3d/4d/5d tensor lengths
# -R <xxx> : reduce dimension ids
# -v <x> :   verification (0=no, 1=yes)
#arg1: data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64, 7: int4)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes)
./bin/example_reduce_blockwise -D 16,64,32,960 -v 1 0 2 1
```

Result
```
./bin/example_reduce_blockwise -D 16,64,32,960 -v 1 0 2 1
launch_and_time_kernel: grid_dim {240, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 0.238063 ms, 264.285 GB/s, DeviceReduceBlockWise<256,M_C4_S1,K_C64_S1,InSrcVectorDim_0_InSrcVectorSize_1_OutDstVectorSize_1>
```

## Run ```example_reduce_multiblock_atomic_add```
```bash
# -D <xxx> : input 3d/4d/5d tensor lengths
# -R <xxx> : reduce dimension ids
# -v <x> :   verification (0=no, 1=yes)
#arg1: data type (0: fp32, 1: fp64)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes)
./bin/example_reduce_multiblock_atomic_add -D 16,64,32,960 -v 1 0 2 0
```

Result
```
./bin/example_reduce_multiblock_atomic_add -D 16,64,32,960 -v 1 0 2 0
Perf: 0 ms, inf GB/s, DeviceReduceMultiBlock<256,M_C4_S1,K_C64_S1,InSrcVectorDim_0_InSrcVectorSize_1_OutDstVectorSize_1>
echo $?
0
```

# Instructions for ```example_reduce_blockwise_two_call```

## Run ```example_reduce_blockwise_two_call```
```bash
#arg1:  verification (0=no, 1=yes(
#arg2:  initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3:  time kernel (0=no, 1=yes)
./bin/example_reduce_blockwise_two_call 1 2 1
```

Result
```
./bin/example_reduce_blockwise_two_call 1 2 1
launch_and_time_kernel: grid_dim {204800, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {6400, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 2.1791 ms, 771.42 GB/s, DeviceReduceBlockWise<256,M_C32_S1,K_C8_S1,InSrcVectorDim_1_InSrcVectorSize_1_OutDstVectorSize_1> => DeviceReduceBlockWise<256,M_C256_S1,K_C1_S1,InSrcVectorDim_1_InSrcVectorSize_1_OutDstVectorSize_1>
```
