# Instructions for ```example_gemm_bilinear_xdl_fp16```

## Run ```example_gemm_bilinear_xdl_fp16```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
#arg4 to 10: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE
#arg11 to 12: alpha, beta
./bin/example_gemm_bilinear_xdl_fp16 1 1 1 3840 4096 4096 4096 4096 4096 4096 0.5 0.5
```
Result (MI100 @ 1502Mhz, 184.6TFlops peak FP16)
```
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
c0_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c0_grid_desc_m_n_{ 3840, 4096}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {480, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 1 times...
Perf: 0.936965 ms, 137.517 TFlops, 102.959 GB/s
error: 0
max_diff: 0, 558.5, 558.5
```
