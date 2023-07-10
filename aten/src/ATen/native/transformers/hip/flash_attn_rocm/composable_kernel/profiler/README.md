## Profile GEMM kernels
```bash
#arg1: tensor operation (gemm=GEMM)
#arg2: data type (0=fp32, 1=fp16)
#arg3: matrix layout (0=NN, 1=NT, 2=TN, 3=TT)
#arg4: verification (0=no, 1=yes)
#arg5: initialization (0=no init, 1=integer value, 2=decimal value)
#arg6: print matrix value (0=no, 1=yes)
#arg7: run kernel # of times (>1)
#arg8 to 13: M, N, K, StrideA, StrideB, StrideC

################        op  datatype  layout  verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC
./bin/ckProfiler      gemm         1       1       1     1    0       5  3840 4096 4096     4096    4096    4096
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```bash
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
....
Best Perf: 1.1933 ms, 107.977 TFlops, 79.0848 GB/s
```

## Profile 2d forward convolution kernels
```bash
#arg1: tensor operation (conv=Convolution)
#arg2: data type (0=fp32, 1=fp16)
#arg3: input tensor layout (0=NCHW, 1=NHWC)
#arg4: weight tensor layout (0=KCYX, 1=KYXC)
#arg5: output tensor layout (0=NKHW, 1=NHWK)
#arg6: verification (0=no, 1=yes)
#arg7: initialization (0=no init, 1=integer value, 2=decimal value)
#arg8: print matrix value (0=no, 1=yes)
#arg9: run kernel # of times (>1)
#arg10 to 24: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, RightPx
 ################          op datatype  in_layout   wei_layout  out_layout  verify  init  log  repeat  N__ K___ C___ Y X Hi__ Wi__ Strides Dilations LeftPads RightPads
 ./bin/ckProfiler  conv2d_fwd        1          1            1           1       1     1    0       5  128  256  192 3 3   71   71     2 2       1 1      1 1       1 1
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
in_n_c_hi_wi: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
wei_k_c_y_x: dim 4, lengths {256, 192, 3, 3}, strides {1728, 1, 576, 192}
out_n_k_ho_wo: dim 4, lengths {128, 256, 36, 36}, strides {331776, 1, 9216, 256}
....
Best Perf: 1.42509 ms, 102.988 TFlops, 234.086 GB/s
```
