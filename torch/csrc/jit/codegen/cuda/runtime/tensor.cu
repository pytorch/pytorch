template <typename T, int N>
struct Tensor {
  __device__ T& operator[](nvfuser_index_t ind) {
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
  __device__ T& operator[](nvfuser_index_t) {
    return *data;
  };

  T* data;
};
