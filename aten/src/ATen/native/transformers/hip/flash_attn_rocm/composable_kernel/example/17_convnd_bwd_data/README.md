# Instructions for ```example_convnd_bwd_data_xdl```

## Run ```example_example_convnd_bwd_data_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4: num_dim_spatial(1|2|3)
#arg5 to ...: N, K, C, [Z,] [Y,] X, [Di,] [Hi,] Wi, S[z,] [Sy,] Sx, [Dz,] [Dy,] Dx, [LeftPz,] [LeftPy,] LeftPx, [RightPy,] [RightPy,] RightPx
./bin/example_convnd_bwd_data_xdl 0 1 5 
```

Result
```
in_n_c_hi_wi: dim 4, lengths {128, 128, 71, 71}, strides {645248, 1, 9088, 128}
wei_k_c_y_x: dim 4, lengths {256, 128, 3, 3}, strides {1152, 1, 384, 128}
out_n_k_ho_wo: dim 4, lengths {128, 256, 36, 36}, strides {331776, 1, 9216, 256}
arg.a_grid_desc_k0_m_k1_container_{128, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{128, 128, 8}
arg.c_grid_desc_m_n_container_{ 175232, 128}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 2, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {1369, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
arg.a_grid_desc_k0_m_k1_container_{64, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{64, 128, 8}
arg.c_grid_desc_m_n_container_{ 175232, 128}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 2, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {1369, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
arg.a_grid_desc_k0_m_k1_container_{64, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{64, 128, 8}
arg.c_grid_desc_m_n_container_{ 175232, 128}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 2, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {1369, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
arg.a_grid_desc_k0_m_k1_container_{32, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{32, 128, 8}
arg.c_grid_desc_m_n_container_{ 175232, 128}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 2, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {1369, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
Perf: 1.40031 ms, 69.8734 TFlops, 179.037 GB/s
```
