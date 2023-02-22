template <typename T, int N>
struct Tensor {
  __device__ T& operator[](nvfuser_index_t ind) {
#ifdef ASSERT_OUT_OF_BOUND
    int64_t max_ind = 0;
#pragma unroll
    for (int i = 0; i < N; i++) {
      max_ind += (size[i] - 1) * stride[i];
    }
    assert(ind >= 0 && ind <= max_ind);
#endif
    return data[ind];
  };

  T* data;
  nvfuser_index_t size[N];
  nvfuser_index_t stride[N];
};

// Specialization for 0-dim case as it does not need size and stride arrays.
// They will be an error as well since zero-length arrays are not allowed.
template <typename T>
struct Tensor<T, 0> {
  __device__ T& operator[](nvfuser_index_t i) {
#ifdef ASSERT_OUT_OF_BOUND
    assert(i == 0);
#endif
    return *data;
  };

  T* data;
};

// Specialization for 0-dim case that's easy to pass in a CPU based tensor.
template <typename T>
struct CpuScalarTensor {
  __device__ T& operator[](int i) {
#ifdef ASSERT_OUT_OF_BOUND
    assert(i == 0);
#endif
    return data;
  };

  T data;
};
