# Instructions for ```example_dual_reduce```

## Run ```example_dual_reduce_multiblock```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes) 
./bin/example_dual_reduce_multiblock -D 600,28,28,256 -v 1 2 1
```

Result
```
./bin/example_dual_reduce_multiblock -D 600,28,28,256 -v 1 2 1                        
launch_and_time_kernel: grid_dim {150, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 1.19529 ms, 201.499 GB/s, DeviceMultipleReduceBlockWise<256,M_C4_S1,K_C64_S1,InSrcVectorDim_1_InSrcVectorSize_1,OutDstVectorSize_1_1>
```

## Run ```example_dual_reduce_threadwise```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes)
./bin/example_dual_reduce_multiblock -D 8000,4,4,4 -v 1 2 1
```

Result
```
./bin/example_dual_reduce_threadwise -D 8000,4,4,4 -v 1 2 1
launch_and_time_kernel: grid_dim {32, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 0.01512 ms, 71.9577 GB/s, DeviceMultipleReduceThreadwise<256,M_C256_S1,K_C1_S4,InSrcVectorDim_1_InSrcVectorSize_2,OutDstVectorSize_1_1>
```
