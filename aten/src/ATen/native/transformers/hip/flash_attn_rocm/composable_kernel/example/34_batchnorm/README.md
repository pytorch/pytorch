# Instructions for ```batchnorm nhwc``` Example

## Run ```batchnorm forward nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1:  data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)
#arg2: 1/0 to indicate whether to update the moving average and variance (0=no, 1=yes)
#arg3: 1/0 to indicate whether to save result mean/invVariance (0=no, 1=yes)
#arg4: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg5: time kernel (0=no, 1=yes) 
./bin/example_batchnorm_forward -D 128,16,16,1024 -v 1 0 0 1 2 1
```

Result 
```
./bin/example_batchnorm_forward -D 128,16,16,1024 -v 1 0 0 1 2 1
launch_and_time_kernel: grid_dim {64, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {120, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {120, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 2.08231 ms, 354.519 GB/s
```

Result
```
./bin/example_batchnorm_forward -D 128,16,16,1024 -v 1 0 1 0 2 0
echo $?
0
```

## Run ```batchnorm infer nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1:  data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes)
./bin/example_batchnorm_infer -D 128,16,16,1024 -v 1 0 2 1
```

Result
```
./bin/example_batchnorm_infer -D 128,16,16,1024 -v 1 0 2 1
launch_and_time_kernel: grid_dim {120, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 1.28235 ms, 523.329 GB/s
```

## Run ```batchnorm backward nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
Arg1: data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)
Arg2 -- 1/0 to indicate whether to use saved mean and invVariance
Arg3 -- init method used for dy and bnScale (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
Arg4 -- time kernel (0=no, 1=yes)
Arg5: use multi-block welford (0=n0, 1=yes)
./bin/example_batchnorm_backward -D 128,16,3,1024 -v 1 0 0 3 1 1
```

Result 
```
./bin/example_batchnorm_backward -D 128,16,3,1024 -v 1 0 0 3 1 1
launch_and_time_kernel: grid_dim {6144, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {6144, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {6144, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 0.411026 ms, 91.8702 GB/s
```
