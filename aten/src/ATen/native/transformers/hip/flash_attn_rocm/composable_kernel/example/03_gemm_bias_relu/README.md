# Instructions for ```example_gemm_bias_relu_xdl_fp16```

## Run ```example_gemm_bias_relu_xdl_fp16```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
#arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideE
./bin/example_gemm_bias_relu_xdl_fp16 1 1 1 3840 4096 4096 4096 4096 4096
```
