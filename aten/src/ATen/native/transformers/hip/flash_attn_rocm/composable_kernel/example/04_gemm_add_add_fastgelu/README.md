# Instructions for ```example_gemm_add_add_fastgelu_xdl_fp16```

## Run ```example_gemm_add_add_fastgelu_xdl_fp16```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
#arg4 to 11: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD0, StrideD1, StrideE"
./bin/example_gemm_add_add_fastgelu_xdl_fp16 1 1 1
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
d0_m_n: dim 2, lengths {3840, 4096}, strides {0, 1}
d1_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
e_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
launch_and_time_kernel: grid_dim {480, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 1.26914 ms, 101.525 TFlops, 100.804 GB/s, DeviceGemmMultipleD_Xdl_CShuffle<256, 256, 128, 32, 8, 8>
```
