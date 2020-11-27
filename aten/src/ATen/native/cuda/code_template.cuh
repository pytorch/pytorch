  typedef long long int int64_t;
  static_assert(sizeof(int64_t) == 8, "expected size does not match");
  constexpr int num_threads = 64;
  constexpr int thread_work_size = 4; //TODO make template substitution once we decide where those vars live
  constexpr int block_work_size = thread_work_size * num_threads;

  template <typename T>
  struct DivMod {
    T div;
    T mod;

    __device__ DivMod(T _div, T _mod) {
      div = _div;
      mod = _mod;
    }
  };

  //<unsigned int>
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

  struct OffsetCalculator {
    OffsetCalculator() = default;
    __device__ void index_to_offset(${index_type} offsets[${nInputs}], ${index_type} linear_idx) const {
      #pragma unroll
      for (int arg = 0; arg < ${nInputs}; ++arg) {
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
        for (int arg = 0; arg < ${nInputs}; ++arg) {
          offsets[arg] += divmod.mod * strides_[dim][arg];
        }
      }
    }

    int dims;
    IntDivider sizes_[25];
    // NOTE: this approach will not support nInputs == 0
    ${index_type} strides_[25][${nInputs}];
  };

  ${functor}

  // NOTE: assumes the op is binary (i.e. has three arguments out, a, and b)
  // TODO: setup grid-stride loop
  extern "C" __global__
  void ${name}_kernel(
      ${name}<${scalar_type}> functor,
      const int numel,
      char* data,
      OffsetCalculator input_calculator,
      OffsetCalculator output_calculator) {

    int remaining = numel - block_work_size * blockIdx.x;

    // NOTE: only the first thread operates on the first element for now
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      // ${scalar_type} a_value;
      // int a_offset = a.index_to_offset(0);

      // ${scalar_type} b_value;
      // int b_offset = b.index_to_offset(0);

      // int out_offset = out.index_to_offset(0);

      // // TODO: refactor the loading, see c10::fetch_and_cast
      // if (a.scalar_type_ == 0) {
      //   a_value = static_cast<${scalar_type}>(*(reinterpret_cast<float*>(a.data_ + a_offset)));
      // } else if (a.scalar_type_ == 1) {
      //   a_value = static_cast<${scalar_type}>(*(reinterpret_cast<double*>(a.data_ + a_offset)));
      // }

      // if (b.scalar_type_ == 0) {
      //   b_value = static_cast<${scalar_type}>(*(reinterpret_cast<float*>(b.data_ + b_offset)));
      // } else if (b.scalar_type_ == 1) {
      //   b_value = static_cast<${scalar_type}>(*(reinterpret_cast<double*>(b.data_ + b_offset)));
      // }

      // ${scalar_type} out_value = functor(a_value, b_value);

      // // TODO: refactor the storing, see c10::cast_and_store
      // if (out.scalar_type_ == 0) {
      //   *(reinterpret_cast<float*>(out.data_ + out_offset)) = static_cast<float>(out_value);
      // } else if (out.scalar_type_ == 1) {
      //   *(reinterpret_cast<double*>(out.data_ + out_offset)) = static_cast<double>(out_value);
      // }

      // printf("%f\n", out_value);
    }
  }

// instantiations here
