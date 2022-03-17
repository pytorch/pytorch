  template <typename T>
  struct DivMod {
  T div;
  T mod;

  __device__ DivMod(T _div, T _mod) {
      div = _div;
      mod = _mod;
  }
  };


  struct IntDivider {
  IntDivider() = default;

  __device__ inline unsigned int div(unsigned int n) const {
  unsigned int t = __umulhi(n, m1);
  return (t + n) >> shift;
  }

  __device__ inline unsigned int mod(unsigned int n) const {
  return n - div(n) * divisor;
  }

  __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
  unsigned int q = div(n);
  return DivMod<unsigned int>(q, n - q * divisor);
  }

  unsigned int divisor;  // d above.
  unsigned int m1;  // Magic number: m' above.
  unsigned int shift;  // Shift amounts.
  };

  template <int NARGS>
  struct TrivialOffsetCalculator {
    // The offset for each argument. Wrapper around fixed-size array.
    // The offsets are in # of elements, not in bytes.
    Array<${index_type}, NARGS> get(${index_type} linear_idx) const {
      Array<${index_type}, NARGS> offsets;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = linear_idx;
      }
      return offsets;
    }
  };

  template<int NARGS>
  struct OffsetCalculator {
  OffsetCalculator() = default;
  __device__ __forceinline__ Array<${index_type}, NARGS> get(${index_type} linear_idx) const {
      Array<${index_type}, NARGS> offsets;
      #pragma unroll
      for (int arg = 0; arg < NARGS; ++arg) {
      offsets[arg] = 0;
      }

      #pragma unroll
      for (int dim = 0; dim < 25; ++dim) {
      if (dim == dims) {
          break;
      }

      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      #pragma unroll
      for (int arg = 0; arg < NARGS; ++arg) {
          offsets[arg] += divmod.mod * strides_[dim][arg];
      }
      //printf("offset calc thread dim size stride offset %d %d %d %d %d %d %d %d\n",
      //threadIdx.x, dim, sizes_[dim].divisor, strides_[dim][0], offsets[0], linear_idx, divmod.div, divmod.mod);
      }
      return offsets;
  }

    int dims;
    IntDivider sizes_[25];
    // NOTE: this approach will not support nInputs == 0
    ${index_type} strides_[25][NARGS];
  };

  #define C10_HOST_DEVICE __host__ __device__
  #define C10_DEVICE __device__

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
    return __shfl_down_sync(mask, value, delta, width);
}


  struct ReduceConfig {
  //has to match host-side ReduceConfig in the eager code
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  static constexpr int input_vec_size = 4;
  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int block_width;
  int block_height;
  int num_threads;

  bool vectorize_input = false;
  int output_vec_size = 1;

  C10_HOST_DEVICE bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  C10_HOST_DEVICE bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

  C10_HOST_DEVICE bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  C10_DEVICE bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
      (!should_block_x_reduce() || threadIdx.x == 0) &&
      (!should_block_y_reduce() || threadIdx.y == 0);
  }

  C10_DEVICE bool should_reduce_tail() const {
    return (!should_block_y_reduce() || threadIdx.y == 0) &&
      (!should_global_reduce() || blockIdx.y == 0);
  }

  C10_HOST_DEVICE int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] +
            warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  template <int output_vec_size>
  C10_HOST_DEVICE int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] +
            warp * output_mult[BLOCK_Y] +
            cta1 * step_output) * output_vec_size;
  }

  C10_DEVICE int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  C10_DEVICE int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }


  };


  ${functor}

  struct func_wrapper_t {
  using scalar_t = ${scalar_type};


  static inline __device__ scalar_t project(scalar_t arg) {
    return (scalar_t) arg;
  }

  static inline __device__ scalar_t warp_shfl_down(scalar_t arg, int offset) {
    return WARP_SHFL_DOWN(arg, offset);
  }

  static __device__ scalar_t translate_idx(scalar_t acc, int64_t /*idx*/) {
    return acc;
  }

  // wrap a normal reduction that ignores the index
 __device__ scalar_t reduce(scalar_t acc, scalar_t val, int64_t idx) const {
     return combine<scalar_t>(acc, val);
   }
  };


  extern "C"
  __global__ void reduction_${name}_kernel(const int numel){

  }
