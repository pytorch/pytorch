# Instructions for ```example_contraction_bilinear_xdl_fp32```

## Run
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
./bin/example_contraction_bilinear_xdl_fp32 1 1 1
```

Result (MI100 @ dynammic freq, 46TFlops peak FP32)
```
a_ms_ks: dim 4, lengths {30, 128, 32, 64}, strides {524288, 4096, 128, 1}
b_ks_ns: dim 4, lengths {32, 64, 32, 64}, strides {128, 1, 524288, 4096}
c_ms_ns: dim 4, lengths {30, 128, 32, 64}, strides {524288, 4096, 128, 1}
launch_and_time_kernel: grid_dim {240, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 0.843286 ms, 38.1985 TFlops, 94.5014 GB/s, DeviceContractionMultipleD_Xdl_CShuffle<256, 256, 128, 16, 4, 4>
```
