# Instructions for ```example_grouped_gemm_xdl```

## Run ```example_grouped_gemm_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
./bin/example_grouped_gemm_xdl_fp16 0 1 5
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
gemm[0] a_m_k: dim 2, lengths {256, 64}, strides {64, 1} b_k_n: dim 2, lengths {64, 128}, strides {1, 64} c_m_n: dim 2, lengths {256, 128}, strides {128, 1}
gemm[1] a_m_k: dim 2, lengths {512, 128}, strides {128, 1} b_k_n: dim 2, lengths {128, 256}, strides {1, 128} c_m_n: dim 2, lengths {512, 256}, strides {256, 1}
gemm[2] a_m_k: dim 2, lengths {768, 192}, strides {192, 1} b_k_n: dim 2, lengths {192, 384}, strides {1, 192} c_m_n: dim 2, lengths {768, 384}, strides {384, 1}
gemm[3] a_m_k: dim 2, lengths {1024, 256}, strides {256, 1} b_k_n: dim 2, lengths {256, 512}, strides {1, 256} c_m_n: dim 2, lengths {1024, 512}, strides {512, 1}
group: 0 arg.a_grid_desc_k0_m_k1_{8, 256, 8}, arg.b_grid_desc_k0_n_k1_{8, 128, 8}, arg.c_grid_desc_m_n_{ 256, 128}
group: 1 arg.a_grid_desc_k0_m_k1_{16, 512, 8}, arg.b_grid_desc_k0_n_k1_{16, 256, 8}, arg.c_grid_desc_m_n_{ 512, 256}
group: 2 arg.a_grid_desc_k0_m_k1_{24, 768, 8}, arg.b_grid_desc_k0_n_k1_{24, 384, 8}, arg.c_grid_desc_m_n_{ 768, 384}
group: 3 arg.a_grid_desc_k0_m_k1_{32, 1024, 8}, arg.b_grid_desc_k0_n_k1_{32, 512, 8}, arg.c_grid_desc_m_n_{ 1024, 512}
launch_and_time_kernel: grid_dim {30, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 5 times...
Perf: 0.037887 ms, 11.0706 TFlops, 90.8132 GB/s, DeviceGroupedGemmXdl<256, 256, 128, 4, 8, 32, 32, 4, 2>
```
