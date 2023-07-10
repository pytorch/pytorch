# Instructions for ```example_convnd_fwd_xdl```

## Run ```example_convnd_fwd_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4: N spatial dimensions (default 2)
#Following arguments (depending on number of spatial dims):
# N, K, C, 
# <filter spatial dimensions>, (ie Y, X for 2D)
# <input image spatial dimensions>, (ie Hi, Wi for 2D)
# <strides>, (ie Sy, Sx for 2D)
# <dilations>, (ie Dy, Dx for 2D)
# <left padding>, (ie LeftPy, LeftPx for 2D)
# <right padding>, (ie RightPy, RightPx for 2D)
./bin/example_convnd_fwd_xdl 0 1 100
```

Result (MI100 @ 1087Mhz, 33.4TFlops peak FP32)
```
input: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
weights: dim 4, lengths {256, 192, 3, 3}, strides {1728, 1, 576, 192}
output: dim 4, lengths {128, 256, 36, 36}, strides {331776, 1, 9216, 256}
arg.a_grid_desc_k0_m_k1_{432, 165888, 4}
arg.b_grid_desc_k0_n_k1_{432, 256, 4}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {1296, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 100 times...
Perf: 4.43736 ms, 33.0753 TFlops, 150.357 GB/s
```
