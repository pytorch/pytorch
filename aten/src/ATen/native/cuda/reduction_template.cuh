namespace at::cuda {
//windows doesn't like large string literals, so split in two
const std::string reduction_template_0 = R"ESCAPE(
  #define C10_HOST_DEVICE __host__ __device__
  #define C10_DEVICE __device__
  #if defined(__clang__) && defined(__HIP__)
  #ifndef __forceinline__
  #define __forceinline__ inline __attribute__((always_inline))
  #endif
  // until ROCm support for kernel asserts is restored
  #define assert(expr) (static_cast<void>(0))
  #endif

  template <typename T>
  __device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
  {
  #if defined(__clang__) && defined(__HIP__)
    return __shfl_down(value, delta, width);
  #else
    return __shfl_down_sync(mask, value, delta, width);
  #endif
  }


  #if ${complex}
  template <typename T>
  __device__ __forceinline__ std::complex<T> WARP_SHFL_DOWN(std::complex<T> value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
  {
    return std::complex<T>(
  #if defined(__clang__) && defined(__HIP__)
        __shfl_down(value.real(), delta, width),
        __shfl_down(value.imag(), delta, width));
  #else
        __shfl_down_sync(mask, value.real(), delta, width),
        __shfl_down_sync(mask, value.imag(), delta, width));
  #endif
  }
  #endif

  // aligned vector generates vectorized load/store on CUDA
  template<typename scalar_t, int vec_size>
  struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
    scalar_t val[vec_size];
  };


  C10_HOST_DEVICE static void reduce_fraction(size_t &numerator, size_t &denominator) {
    // get GCD of num and denom using Euclid's algorithm.
    // Can replace this with std::gcd if we ever support c++17.
    size_t a = denominator;
    size_t b = numerator;
    while (b != 0) {
        a %= b;
        // swap(a,b)
        size_t tmp = a;
        a = b;
        b = tmp;
    }

    // a is now the GCD
    numerator /= a;
    denominator /= a;
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


//TODO this will need to be different for more generic reduction functions
namespace reducer {

  using scalar_t = ${scalar_type};
  using arg_t = ${reduction_accum_type};
  using out_scalar_t = ${result_type};


  inline __device__ ${functor}

  inline __device__ out_scalar_t project(arg_t arg) {
    return (out_scalar_t) arg;
  }

  inline __device__ arg_t warp_shfl_down(arg_t arg, int offset) {
    return WARP_SHFL_DOWN(arg, offset);
  }

  inline __device__ arg_t translate_idx(arg_t acc, int64_t /*idx*/) {
    return acc;
  }

  // wrap a normal reduction that ignores the index
  inline __device__ arg_t reduce(arg_t acc, arg_t val, int64_t idx) {
     return combine(acc, val);
  }
}


struct ReduceJitOp {
  using scalar_t = ${scalar_type};
  using arg_t = ${reduction_accum_type};
  using out_scalar_t = ${result_type};

  using InputCalculator = OffsetCalculator<1>;
  using OutputCalculator = OffsetCalculator<2>;

//   static constexpr bool can_accumulate_in_output =
//     std::is_convertible_v<arg_t, out_scalar_t>
//     && std::is_convertible_v<out_scalar_t, arg_t>;

  static constexpr int input_vec_size = ReduceConfig::input_vec_size;

  arg_t ident;
  ReduceConfig config;
  InputCalculator input_calc;
  OutputCalculator output_calc;
  const void* src;
  const char* dst[2]; //it accepts at most two destinations
  // acc_buf used for accumulation among sub Tensor Iterator when accumulation on
  // output is not permissible
  void* acc_buf;
  // cta_buf used for accumulation between blocks during global reduction
  void* cta_buf;
  int* semaphores;
  int64_t base_idx;
  bool accumulate;
  bool final_output;
  int noutputs;


  C10_DEVICE void run() const {
    extern __shared__ char shared_memory[];
    uint32_t output_idx = config.output_idx<${output_vec_size}>();
    uint32_t input_idx = config.input_idx();
    auto base_offsets1 = output_calc.get(output_idx)[1];

    using arg_vec_t = Array<arg_t, ${output_vec_size}>;
    arg_vec_t value;

    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      const scalar_t* input_slice = (const scalar_t*)((const char*)src + base_offsets1);

      value = thread_reduce<${output_vec_size}>(input_slice);
    }

    if (config.should_block_y_reduce()) {
      value = block_y_reduce<${output_vec_size}>(value, shared_memory);
    }
    if (config.should_block_x_reduce()) {
      value = block_x_reduce<${output_vec_size}>(value, shared_memory);
    }

    using out_ptr_vec_t = Array<out_scalar_t*, ${output_vec_size}>;
    using offset_vec_t = Array<uint32_t, ${output_vec_size}>;
    offset_vec_t base_offsets;
    out_ptr_vec_t out;

    #pragma unroll
    for (int i = 0; i < ${output_vec_size}; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = (out_scalar_t*)((char*)dst[0] + base_offsets[i]);
    }

    arg_vec_t* acc = nullptr;
    if (acc_buf != nullptr) {
      size_t numerator = sizeof(arg_t);
      size_t denominator = sizeof(out_scalar_t);
      reduce_fraction(numerator, denominator);
      acc = (arg_vec_t*)((char*)acc_buf + (base_offsets[0] * numerator / denominator));
    }

    if (config.should_global_reduce()) {
      value = global_reduce<${output_vec_size}>(value, acc, shared_memory);
    } else if (config.should_store(output_idx)) {
      if (accumulate) {
        #pragma unroll
        for (int i = 0; i < ${output_vec_size}; i++) {
          value[i] = reducer::translate_idx(value[i], base_idx);
        }
      }

      if (acc == nullptr) {
        if (accumulate) {
          value = accumulate_in_output<${output_vec_size}>(out, value);
        }
        if (final_output) {
          set_results_to_output<${output_vec_size}>(value, base_offsets);
        } else {
          #pragma unroll
          for (int i = 0; i < ${output_vec_size}; i++) {
            *(out[i]) = get_accumulated_output(out[i], value[i]);
          }
        }
      } else {
        if (accumulate) {
          #pragma unroll
          for (int i = 0; i < ${output_vec_size}; i++) {
            value[i] = reducer::combine((*acc)[i], value[i]);
          }
        }
        if (final_output) {
          set_results_to_output<${output_vec_size}>(value, base_offsets);
        } else {
          *acc = value;
        }
      }
    }
  }

  template <int output_vec_size>
  C10_DEVICE Array<arg_t, output_vec_size> thread_reduce(const scalar_t* data) const {
    if (config.vectorize_input) {
      assert(output_vec_size == 1);
      // reduce at the header of input_slice where memory is not aligned,
      // so that thread_reduce will have an aligned memory to work on.
      return {input_vectorized_thread_reduce_impl(data)};
    } else {
      uint32_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
      bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
      if (is_contiguous) {
        return thread_reduce_impl<output_vec_size>(data, [](uint32_t idx) { return idx; });
      } else if (input_calc.dims == 1) {
        return thread_reduce_impl<output_vec_size>(data, [&](uint32_t idx) { return idx * element_stride; });
      } else {
        return thread_reduce_impl<output_vec_size>(data, [&](uint32_t idx) { return input_calc.get(idx)[0] / sizeof(scalar_t); });
      }
    }
  }

  C10_DEVICE arg_t input_vectorized_thread_reduce_impl(const scalar_t* data) const {
    uint32_t end = config.num_inputs;

    // Handle the head of input slice where data is not aligned
    arg_t value = ident;
    constexpr int align_bytes = alignof(aligned_vector<scalar_t, input_vec_size>);
    constexpr int align_elements = align_bytes / sizeof(scalar_t);
    int shift = ((int64_t)data) % align_bytes / sizeof(scalar_t);
    if (shift > 0) {
      data -= shift;
      end += shift;
      if(threadIdx.x >= shift && threadIdx.x < align_elements && config.should_reduce_tail()){
        value = reducer::reduce(value, data[threadIdx.x], threadIdx.x - shift);
      }
      end -= align_elements;
      data += align_elements;
      shift = align_elements - shift;
    }

    // Do the vectorized reduction
    using load_t = aligned_vector<scalar_t, input_vec_size>;

    uint32_t idx = config.input_idx();
    const uint32_t stride = config.step_input;

    // Multiple accumulators to remove dependency between unrolled loops.
    arg_t value_list[input_vec_size];
    value_list[0] = value;

    #pragma unroll
    for (int i = 1; i < input_vec_size; i++) {
      value_list[i] = ident;
    }

    scalar_t values[input_vec_size];

    load_t *values_vector = reinterpret_cast<load_t*>(&values[0]);

    while (idx * input_vec_size + input_vec_size - 1 < end) {
      *values_vector = reinterpret_cast<const load_t*>(data)[idx];
      #pragma unroll
      for (uint32_t i = 0; i < input_vec_size; i++) {
        value_list[i] = reducer::reduce(value_list[i], values[i], shift + idx * input_vec_size + i);
      }
      idx += stride;
    }

    // tail
    uint32_t tail_start = end - end % input_vec_size;
    if (config.should_reduce_tail()) {
      int idx = tail_start + threadIdx.x;
      if (idx < end) {
        value_list[0] = reducer::reduce(value_list[0], data[idx], idx + shift);
      }
    }

    // combine accumulators
    #pragma unroll
    for (int i = 1; i < input_vec_size; i++) {
      value_list[0] = reducer::combine(value_list[0], value_list[i]);
    }
    return value_list[0];
  }

  template <int output_vec_size, typename offset_calc_t>
  C10_DEVICE Array<arg_t, output_vec_size> thread_reduce_impl(const scalar_t* data_, offset_calc_t calc) const {
    uint32_t idx = config.input_idx();
    const uint32_t end = config.num_inputs;
    const uint32_t stride = config.step_input;
    const int vt0=${vt0};

    using arg_vec_t = Array<arg_t, output_vec_size>;
    using load_t = aligned_vector<scalar_t, output_vec_size>;
    const load_t* data = reinterpret_cast<const load_t*>(data_);

    // Multiple accumulators to remove dependency between unrolled loops.
    arg_vec_t value_list[vt0];

    #pragma unroll
    for (int i = 0; i < vt0; i++) {
      #pragma unroll
      for (int j = 0; j < output_vec_size; j++) {
        value_list[i][j] = ident;
      }
    }

    load_t values[vt0];

    while (idx + (vt0 - 1) * stride < end) {
      #pragma unroll
      for (uint32_t i = 0; i < vt0; i++) {
        values[i] = data[calc(idx + i * stride) / output_vec_size];
      }
      #pragma unroll
      for (uint32_t i = 0; i < vt0; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < output_vec_size; j++) {
          value_list[i][j] = reducer::reduce(value_list[i][j], values[i].val[j], idx + i * stride);
        }
      }
      idx += stride * vt0;
    }

    // tail
    int idx_ = idx;
    #pragma unroll
    for (uint32_t i = 0; i < vt0; i++) {
      if (idx >= end) {
        break;
      }
      values[i] = data[calc(idx) / output_vec_size];
      idx += stride;
    }
    idx = idx_;
    #pragma unroll
    for (uint32_t i = 0; i < vt0; i++) {
      if (idx >= end) {
        break;
      }
      #pragma unroll
      for (uint32_t j = 0; j < output_vec_size; j++) {
        value_list[i][j] = reducer::reduce(value_list[i][j], values[i].val[j], idx);
      }
      idx += stride;
    }

    // combine accumulators
    #pragma unroll
    for (int i = 1; i < vt0; i++) {
      #pragma unroll
      for (uint32_t j = 0; j < output_vec_size; j++) {
        value_list[0][j] = reducer::combine(value_list[0][j], value_list[i][j]);
      }
    }
    return value_list[0];
  }
  template <int output_vec_size>
  C10_DEVICE Array<arg_t, output_vec_size> block_x_reduce(Array<arg_t, output_vec_size> value, char* shared_memory) const {
    using args_vec_t = Array<arg_t, output_vec_size>;
    int dim_x = blockDim.x;
    args_vec_t* shared = (args_vec_t*)shared_memory;
    if (dim_x > warpSize) {
      int address_base = threadIdx.x + threadIdx.y*blockDim.x;
      shared[address_base] = value;
      for (int offset = dim_x/2; offset >= warpSize; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
          args_vec_t other = shared[address_base + offset];
          #pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = reducer::combine(value[i], other[i]);
          }
          shared[address_base] = value;
        }
      }
      dim_x = warpSize;
    }

    __syncthreads();

    for (int offset = 1; offset < dim_x; offset <<= 1) {
      #pragma unroll
      for (int i = 0; i < output_vec_size; i++) {
        arg_t other = reducer::warp_shfl_down(value[i], offset);
        value[i] = reducer::combine(value[i], other);
      }
    }
    return value;
  }

  template <int output_vec_size>
  C10_DEVICE Array<arg_t, output_vec_size> block_y_reduce(Array<arg_t, output_vec_size> value, char* shared_memory) const {
    using args_vec_t = Array<arg_t, output_vec_size>;
    args_vec_t* shared = (args_vec_t*)shared_memory;
    shared[config.shared_memory_offset(0)] = value;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
        args_vec_t other = shared[config.shared_memory_offset(offset)];
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          value[i] = reducer::combine(value[i], other[i]);
        }
        shared[config.shared_memory_offset(0)] = value;
      }
    }
    return value;
  }
  )ESCAPE";

  const std::string reduction_template_1 = R"ESCAPE(

  C10_DEVICE bool mark_block_finished() const {
    __shared__ bool is_last_block_done_shared;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
    }

    __syncthreads();

    return is_last_block_done_shared;
  }

  template <int output_vec_size>
  C10_DEVICE Array<arg_t, output_vec_size> accumulate_in_output(
    Array<out_scalar_t*, output_vec_size> out,
    Array<arg_t, output_vec_size> value
  ) const {
    Array<arg_t, output_vec_size> ret;
    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      ret[i] = reducer::combine(*(out[i]), value[i]);
    }
    return ret;
  }


  C10_DEVICE out_scalar_t get_accumulated_output(
    out_scalar_t* out, arg_t value
  ) const {
    assert(!final_output);
    return (out_scalar_t)value;
  }

  template<class T>
  C10_DEVICE void set_results(const T x, const uint32_t base_offset) const {
    assert(noutputs == 1);
    auto res = (out_scalar_t*)((char*)dst[0] + base_offset);
    *res = x;
  }

//TODO - multi-output reduction - we won't be able to use thrust::pair
//just explicitly specify typed output reads/writes
//Currently implemented for max of two outputs
//   template<class T1, class T2>
//   C10_DEVICE void set_results(const thrust::pair<T1, T2> x, const index_t base_offset) const {
//     if (noutputs >= 1) {
//       auto res0 = (T1*)((char*)dst[0] + base_offset);
//       *res0 = x.first;
//     }
//     if (noutputs >= 2) {
//       // base offset is computed assuming element size being sizeof(T1), so we need to make a
//       // correction to obtain the correct base offset
//       auto res1 = (T2*) ((char *) dst[1] + base_offset / sizeof(T1) * sizeof(T2));
//       *res1 = x.second;
//     }
//   }

  template <int output_vec_size>
  C10_DEVICE void set_results_to_output(Array<arg_t, output_vec_size> value, Array<uint32_t, output_vec_size> base_offset) const {
    assert(final_output);
    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      set_results(reducer::project(value[i]), base_offset[i]);
    }
  }

  template <int output_vec_size>
  C10_DEVICE Array<arg_t, output_vec_size> global_reduce(Array<arg_t, output_vec_size> value, Array<arg_t, output_vec_size> *acc, char* shared_memory) const {
    using arg_vec_t = Array<arg_t, output_vec_size>;
    using out_ptr_vec_t = Array<out_scalar_t*, output_vec_size>;
    using offset_vec_t = Array<uint32_t, output_vec_size>;

    arg_vec_t* reduce_buffer = (arg_vec_t*)cta_buf;
    uint32_t output_idx = config.output_idx<output_vec_size>();
    offset_vec_t base_offsets;
    out_ptr_vec_t out;

    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = (out_scalar_t*)((char*)dst[0] + base_offsets[i]);
    }

    bool should_store = config.should_store(output_idx);
    if (should_store) {
      uint32_t offset = config.staging_memory_offset(blockIdx.y);
      reduce_buffer[offset] = value;
    }

    __threadfence(); // make sure writes are globally visible
    __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done
    bool is_last_block_done = mark_block_finished();

    if (is_last_block_done) {
      __threadfence(); //complete acquire pattern
      value = ident;
      if (config.should_block_x_reduce()) {
        uint32_t input_offset = threadIdx.x + threadIdx.y * blockDim.x;
        uint32_t step = blockDim.x * blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          uint32_t idx = config.staging_memory_offset(input_offset);
          arg_vec_t next = reduce_buffer[idx];
          #pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = reducer::combine(value[i], next[i]);
          }
        }
      } else {
        uint32_t input_offset = threadIdx.y;
        uint32_t step = blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          uint32_t idx = config.staging_memory_offset(input_offset);
          arg_vec_t next = reduce_buffer[idx];
          #pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = reducer::combine(value[i], next[i]);
          }
        }
      }
      value = block_y_reduce(value, shared_memory);
      if (config.should_block_x_reduce()) {
        value = block_x_reduce<output_vec_size>(value, shared_memory);
      }
      if (should_store) {
        if (accumulate) {
          #pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = reducer::translate_idx(value[i], base_idx);
          }
        }

        if (acc == nullptr) {
          if (accumulate) {
            value = accumulate_in_output<output_vec_size>(out, value);
          }
          if (final_output) {
            set_results_to_output<output_vec_size>(value, base_offsets);
          } else {
            #pragma unroll
            for (int i = 0; i < output_vec_size; i++) {
              *(out[i]) = get_accumulated_output(out[i], value[i]);
            }
          }
        } else {
          if (accumulate) {
            #pragma unroll
            for (int i = 0; i < output_vec_size; i++) {
              value[i] = reducer::combine((*acc)[i], value[i]);
            }
          }
          if (final_output) {
            set_results_to_output<output_vec_size>(value, base_offsets);
          } else {
            *acc = value;
          }
        }
      }
    }

    return value;
  }
};

extern "C"
__launch_bounds__(${max_threads_lb}, 4)
__global__ void reduction_${name}_kernel(ReduceJitOp r){
  r.run();
}
)ESCAPE";

const std::string reduction_template = reduction_template_0 + reduction_template_1;


const std::string &get_reduction_template() {
  return reduction_template;
}

} // namespace at::cuda
