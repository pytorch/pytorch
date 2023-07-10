# Instructions for ```example_pool2d_fwd``` Examples

## Run ```example_pool2d_fwd_fp16```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes)
#arg4 to 15: N, C, Y, X, Hi, Wi, Sy, Sx, LeftPy, LeftPx, RightPy, RightPx
./bin/example_pool2d_fwd_fp16 1 1 1
```

Result 
```
in_n_c_hi_wi: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
out_n_c_ho_wo: dim 4, lengths {128, 192, 36, 36}, strides {248832, 1, 6912, 192}
launch_and_time_kernel: grid_dim {124416, 1, 1}, block_dim {64, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 0.397436 ms, 1.44252 TFlops, 783.713 GB/s
```

## Run ```example_pool2d_fwd_fp32```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes)
#arg4 to 15: N, C, Y, X, Hi, Wi, Sy, Sx, LeftPy, LeftPx, RightPy, RightPx
./bin/example_pool2d_fwd_fp32 1 1 1
```


Result 
```
./bin/example_pool2d_fwd_fp32 1 1 1
in_n_c_hi_wi: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
out_n_c_ho_wo: dim 4, lengths {128, 192, 36, 36}, strides {248832, 1, 6912, 192}
launch_and_time_kernel: grid_dim {124416, 1, 1}, block_dim {64, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 1.01823 ms, 0.563045 TFlops, 611.8 GB/s
```
