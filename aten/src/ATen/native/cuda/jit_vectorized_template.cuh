typedef long long int int64_t;
typedef unsigned int uint32_t;
typedef signed char int8_t;
typedef char uint8_t;
typedef short int16_t;
static_assert(sizeof(int64_t) == 8, "expected size does not match");
static_assert(sizeof(uint32_t) == 4, "expected size does not match");
static_assert(sizeof(int8_t) == 1, "expected size does not match");
constexpr int num_threads = 64;
constexpr int thread_work_size = 4; //TODO make template substitution once we decide where those vars live
constexpr int block_work_size = thread_work_size * num_threads;
#define ERROR_UNSUPPORTED_CAST assert(false);


// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
// Note, some types have ctype as void because we don't support them in codegen
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
_(uint8_t, Byte) /* 0 */                               \
_(int8_t, Char) /* 1 */                                \
_(int16_t, Short) /* 2 */                              \
_(int, Int) /* 3 */                                    \
_(int64_t, Long) /* 4 */                               \
_(void, Half) /* 5 */                              \
_(float, Float) /* 6 */                                \
_(double, Double) /* 7 */                              \
_(void, ComplexHalf) /* 8 */        \
_(void, ComplexFloat) /* 9 */           \
_(void, ComplexDouble) /* 10 */        \
_(bool, Bool) /* 11 */                                 \
_(void, QInt8) /* 12 */                          \
_(void, QUInt8) /* 13 */                        \
_(void, QInt32) /* 14 */                        \
_(void, BFloat16) /* 15 */                     \
_(void, QUInt4x2) /* 16 */

#define AT_FORALL_SCALAR_TYPES(_) \
_(uint8_t, Byte)                \
_(int8_t, Char)                 \
_(int16_t, Short)               \
_(int, Int)                     \
_(int64_t, Long)                \
_(float, Float)                 \
_(double, Double)

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ENUM)
#undef DEFINE_ENUM
    Undefined,
NumOptions
};


template <typename T, int size>
struct Array {
T data[size];

__device__ T operator[](int i) const {
    return data[i];
}
__device__ T& operator[](int i) {
    return data[i];
}
Array() = default;
Array(const Array&) = default;
Array& operator=(const Array&) = default;
};

struct LoadWithoutCast {
template <typename scalar_t>
__device__ scalar_t load(char* base_ptr, uint32_t offset, int arg=0) {
    return *(reinterpret_cast<scalar_t*>(base_ptr) + offset);
}
};

struct StoreWithoutCast {
template<typename scalar_t>
__device__ void store(scalar_t value, char *base_ptr, uint32_t offset) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
}
};

// aligned vector generates vectorized load/store on CUDA
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};






${functor}

// TODO: setup grid-stride loop

extern "C" __global__
void ${name}_vectorized_kernel(
    const int N,
    Array<char*, ${nInputs}+1> data) //[${nInputs}+1],
    {
    constexpr int vec_size = ${vec_size};
    int remaining = N - block_work_size * blockIdx.x;
    auto thread_idx = threadIdx.x;
    int idx = blockIdx.x;

    if (remaining < block_work_size) {
        assert("not ready yet!");
    } else {
      ${declare_load_arrays}
      ${declare_store_arrays}
      static constexpr int loop_size = thread_work_size / vec_size;
//actual loading
      using vec_t_input = aligned_vector<${scalar_type}, vec_size>;
      ${vector_pointers}
      #pragma unroll
      for (int i = 0; i<loop_size; i++){
        vec_t_input v;
        ${load_vectorized_inputs}
        thread_idx += num_threads;
      }


      #pragma unroll
      for (int j = 0; j < thread_work_size; j++) {
        out[j] = ${name}<${scalar_type}>(${args});
      }
      using vec_t_output = aligned_vector<${result_type}, vec_size>;
      vec_t_output * to_ = reinterpret_cast<vec_t_output *>(data[0]) + block_work_size / vec_size * idx;
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i<loop_size; i++){
        vec_t_output v;
        #pragma unroll
        for (int j=0; j<vec_size; j++){
          v.val[j] = out[vec_size * i + j];
        }
        to_[thread_idx] = v;
        thread_idx += num_threads;
      }


    }



}
