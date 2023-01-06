__global__ void kernel1(Tensor<float, 3> T0, Tensor<float, 3> T1, Tensor<float, 3> T2) {
  int i76;
  i76 = ((((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x)) / (T0.size[1] * T0.size[2]);
  int i78;
  i78 = (((((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x)) % (T0.size[1] * T0.size[2])) / T0.size[2];
  int i79;
  i79 = (((((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x)) % (T0.size[1] * T0.size[2])) % T0.size[2];
  int i120;
  i120 = (((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x);
  if ((i120 < (T0.size[0] * (T0.size[1] * T0.size[2])))) {
    float T4[1];
    T4[0] = 0;
    T4[0]
       = T1[(i76 * T1.stride[0]) + (i78 * T1.stride[1]) + (i79 * T1.stride[2])];
    float T3[1];
    T3[0] = 0;
    T3[0]
       = T0[(i76 * T0.stride[0]) + (i78 * T0.stride[1]) + (i79 * T0.stride[2])];
    float T5[1];
    T5[0]
      = T3[0]
      + T4[0];
    T2[i120]
       = T5[0];
  }
}
