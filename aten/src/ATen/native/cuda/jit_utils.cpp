#define TORCH_ASSERT_NO_OPERATORS
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <c10/util/hash.h>
#include <optional>
#include <ATen/jit_macros.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/code_template.h>
#include <ATen/OpMathType.h>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/cuda/llvm_jit_strings.h>
#include <ATen/native/cuda/reduction_template.cuh>

#include <sstream>
#include <fstream>
#include <cstdio>
#include <iterator> // istreambuf_iterator
#include <cstdlib>
#include <string>

// TODO: C++17 has the filesystem header, which may replace these
#ifdef _WIN32
  // On Windows, the POSIX implementations are considered deprecated. We simply map to the newer variant.
  #include <process.h>
  #include <direct.h>
  #include <io.h>
  #define access _access
  #define getpid _getpid
  #define R_OK    4
  #define W_OK    2
  #define F_OK    0
#else
  #include <sys/types.h>
  #include <sys/stat.h> // mkdir
  #include <unistd.h>
#endif


namespace at::cuda::jit {

// hiprtc already includes some traits, so this removes duplicate definitions of
// integral_constant, is_same, is_integral, enable_if, is_floating_point, is_arithmetic.
// Copied from aten/src/ATen/cuda/llvm_basic.cpp, then modified as above.
// If not compiling for ROCm, return the original get_traits_string().
std::string get_traits_string_but_hiprtc_safe() {
#ifdef USE_ROCM
    return R"ESCAPE(
namespace std {

template <class _Tp>
_Tp&& __declval(int);
template <class _Tp>
_Tp __declval(long);
template <class _Tp>
decltype(__declval<_Tp>(0)) declval() noexcept;

template <class _Tp> struct remove_const            {typedef _Tp type;};
template <class _Tp> struct remove_const<const _Tp> {typedef _Tp type;};
template <class _Tp> using remove_const_t = typename remove_const<_Tp>::type;

template <class _Tp> struct remove_volatile               {typedef _Tp type;};
template <class _Tp> struct remove_volatile<volatile _Tp> {typedef _Tp type;};
template <class _Tp> using remove_volatile_t = typename remove_volatile<_Tp>::type;

template <class _Tp> struct remove_cv
{typedef typename remove_volatile<typename remove_const<_Tp>::type>::type type;};
template <class _Tp> using remove_cv_t = typename remove_cv<_Tp>::type;

template <class _Tp> struct __libcpp_is_floating_point              : public false_type {};
template <>          struct __libcpp_is_floating_point<float>       : public true_type {};
template <>          struct __libcpp_is_floating_point<double>      : public true_type {};
template <>          struct __libcpp_is_floating_point<long double> : public true_type {};

template <class _Tp>
inline constexpr bool is_arithmetic_v = is_arithmetic<_Tp>::value;

template <class _Tp>
struct __numeric_type
{
   static void __test(...);
   static float __test(float);
   static double __test(char);
   static double __test(int);
   static double __test(unsigned);
   static double __test(long);
   static double __test(unsigned long);
   static double __test(long long);
   static double __test(unsigned long long);
   static double __test(double);
   static long double __test(long double);

   typedef decltype(__test(declval<_Tp>())) type;
   static const bool value = !is_same<type, void>::value;
};

template <>
struct __numeric_type<void>
{
   static const bool value = true;
};

// __promote

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&
                 __numeric_type<_A2>::value &&
                 __numeric_type<_A3>::value>
class __promote_imp
{
public:
    static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
    typedef typename __promote_imp<_A3>::type __type3;
public:
    typedef decltype(__type1() + __type2() + __type3()) type;
    static const bool value = true;
};

template <class _A1, class _A2>
class __promote_imp<_A1, _A2, void, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
public:
    typedef decltype(__type1() + __type2()) type;
    static const bool value = true;
};

template <class _A1>
class __promote_imp<_A1, void, void, true>
{
public:
    typedef typename __numeric_type<_A1>::type type;
    static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

} // namespace std
)ESCAPE";
#else
    return get_traits_string();
#endif
}

#ifdef USE_ROCM
const std::string jit_preamble = R"ESCAPE(
#pragma clang force_cuda_host_device begin
)ESCAPE";
const std::string jit_epilogue = R"ESCAPE(
#pragma clang force_cuda_host_device end
)ESCAPE";
#else
const std::string jit_preamble;
const std::string jit_epilogue;
#endif

const std::string jit_common_types = R"ESCAPE(
  #ifdef __HIPCC__
  #define ERROR_UNSUPPORTED_CAST ;
  // corresponds to aten/src/ATen/native/cuda/thread_constants.h
  #define CUDA_OR_ROCM_NUM_THREADS 256
  // corresponds to aten/src/ATen/cuda/detail/OffsetCalculator.cuh
  #define MAX_DIMS 16
  #ifndef __forceinline__
  #define __forceinline__ inline __attribute__((always_inline))
  #endif
  #else
  //TODO use _assert_fail, because assert is disabled in non-debug builds
  #define ERROR_UNSUPPORTED_CAST assert(false);
  #define CUDA_OR_ROCM_NUM_THREADS 128
  #define MAX_DIMS 25
  #endif
  #define POS_INFINITY __int_as_float(0x7f800000)
  #define INFINITY POS_INFINITY
  #define NEG_INFINITY __int_as_float(0xff800000)
  #define NAN __int_as_float(0x7fffffff)

  typedef long long int int64_t;
  typedef unsigned int uint32_t;
  typedef signed char int8_t;
  typedef unsigned char uint8_t;  // NOTE: this MUST be "unsigned char"! "char" is equivalent to "signed char"
  typedef short int16_t;
  static_assert(sizeof(int64_t) == 8, "expected size does not match");
  static_assert(sizeof(uint32_t) == 4, "expected size does not match");
  static_assert(sizeof(int8_t) == 1, "expected size does not match");
  constexpr int num_threads = CUDA_OR_ROCM_NUM_THREADS;
  constexpr int thread_work_size = ${thread_work_size}; // TODO: make template substitution once we decide where those vars live
  constexpr int block_work_size = thread_work_size * num_threads;

  ${traits_string}
  ${cmath_string}

  // NB: Order matters for this macro; it is relied upon in
  // _promoteTypesLookup and the serialization format.
  // Note, some types have ctype as void because we don't support them in codegen
  #define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(at::Half, Half) /* 5 */                                  \
  _(float, Float) /* 6 */                                \
  _(double, Double) /* 7 */                              \
  _(std::complex<at::Half>, ComplexHalf) /* 8 */        \
  _(std::complex<float>, ComplexFloat) /* 9 */                          \
  _(std::complex<double>, ComplexDouble) /* 10 */                         \
  _(bool, Bool) /* 11 */                                 \
  _(void, QInt8) /* 12 */                          \
  _(void, QUInt8) /* 13 */                        \
  _(void, QInt32) /* 14 */                        \
  _(at::BFloat16, BFloat16) /* 15 */                             \

  #define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_QINT(_)       \
  _(uint8_t, Byte)                                                 \
  _(int8_t, Char)                                                  \
  _(int16_t, Short)                                                \
  _(int, Int)                                                      \
  _(int64_t, Long)                                                 \
  _(at::Half, Half)                                                \
  _(float, Float)                                                  \
  _(double, Double)                                                \
  _(std::complex<at::Half>, ComplexHalf)                           \
  _(std::complex<float>, ComplexFloat)                             \
  _(std::complex<double>, ComplexDouble)                           \
  _(bool, Bool)                                                    \
  _(at::BFloat16, BFloat16)


  enum class ScalarType : int8_t {
  #define DEFINE_ENUM(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ENUM)
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
  __device__ Array(T x) {
    for (int i = 0; i < size; i++) {
      data[i] = x;
    }
  }
  };

  ${half_string}
  ${bfloat16_string}
  ${complex_body_string}
  ${complex_half_body_string}
  ${complex_math_string}


)ESCAPE";

//we need to include half, bfloat16 and complex strings to all kernels with half arguments and to all kernels with type casting
//regardless of whether they have half arguments (because fetch_and_cast and cast_and_store loop over all types)
const std::string jiterator_half_support_literal = R"ESCAPE(
namespace at {
struct alignas(2) Half {
  unsigned short x;

  Half() = default;
  inline __host__ __device__ Half(float value){
#ifdef __HIPCC__
    x = __half_as_short(__float2half(value));
#else
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(x) : "f"(value));
#endif
  }
  inline __host__ __device__ operator float() const{
#ifdef __HIPCC__
      return __half2float(*reinterpret_cast<const __half*>(&x));
#else
      float val;
      asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(x)); // do we need const cast here?
      //asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(x)));
      return val;
#endif
  }
};
}
)ESCAPE";

const std::string jiterator_bfloat16_support_literal = R"ESCAPE(
namespace at {
struct alignas(2) BFloat16 {
  unsigned short x;

  __device__ unsigned short __internal_float2bfloat16(
      const float f,
      unsigned int& sign,
      unsigned int& remainder) {
    unsigned int x;

    x = __float_as_uint(f);

    if ((x & 0x7fffffffU) > 0x7f800000U) {
      sign = 0U;
      remainder = 0U;
      return static_cast<unsigned short>(0x7fffU);
    }
    sign = x >> 31;
    remainder = x << 16;
    return static_cast<unsigned short>(x >> 16);
  }


  BFloat16() = default;
  inline __host__ __device__ BFloat16(float value){
  #if __CUDA_ARCH__ >= 800
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n" : "=h"(x) : "f"(value));
  )ESCAPE"
  R"ESCAPE(
  #else
  unsigned int sign;
  unsigned int remainder;
  x = __internal_float2bfloat16(value, sign, remainder);
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((x & 0x1U) != 0U))) {
    x++;
  }
  #endif
  }

  inline __host__ __device__ operator float() const{
#ifdef __HIPCC__
    union
    {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(x) << 16};
    return u.fp32;
#else
    float val;
    asm("{ mov.b32 %0, {0,%1};}\n" : "=f"(val) : "h"(x)); //do we need const cast here?
    return val;
#endif
  }

};
}
)ESCAPE";

// From c10/util/Load.h
const std::string load_support_literal = R"ESCAPE(

  namespace c10 {
    template <typename T>
    struct LoadImpl {
      __device__ static T apply(const void *src) {
        return *reinterpret_cast<const T*>(src);
      }
    };

    template <>
    struct LoadImpl<bool> {
      __device__ static bool apply(const void *src) {
        static_assert(sizeof(bool) == sizeof(char), "");
        return LoadImpl<char>::apply(src);
      }
    };

    template <typename T>
    __device__ T load(const void *src) {
      return LoadImpl<T>::apply(src);
    }

    template <typename scalar_t>
    __device__ scalar_t load(const scalar_t *src) {
      return LoadImpl<scalar_t>::apply(src);
    }
  }  // namespace c10

)ESCAPE";

// copy-pasted from c10/util/TypeCast.h and c10/core/DynamicCast.h
const std::string dynamic_cast_support_literal = R"ESCAPE(

  template <typename T>
  struct is_complex : public std::false_type {};

  template <typename T>
  struct is_complex<std::complex<T>> : public std::true_type {};

  template <typename dest_t, typename src_t>
  struct needs_real {
    constexpr static bool value =
        (is_complex<src_t>::value && !is_complex<dest_t>::value);
  };

  template <bool, typename src_t>
  struct maybe_real {
    static inline src_t apply(src_t src) {
      return src;
    }
  };

  template <typename src_t>
  struct maybe_real<true, src_t> {
    static inline decltype(auto) apply(src_t src) {
      return src.real();
    }
  };

  template <typename dest_t, typename src_t>
  struct static_cast_with_inter_type {
    static inline dest_t apply(
        src_t src) {
      constexpr bool real = needs_real<dest_t, src_t>::value;
      return static_cast<dest_t>(maybe_real<real, src_t>::apply(src));
    }
  };

  template <typename src_t>
  struct static_cast_with_inter_type<uint8_t, src_t> {
    static inline uint8_t apply(
        src_t src) {
      constexpr bool real = needs_real<uint8_t, src_t>::value;
      return static_cast<uint8_t>(
          static_cast<int64_t>(maybe_real<real, src_t>::apply(src)));
    }
  };

  template <>
  struct static_cast_with_inter_type<std::complex<at::Half>, at::BFloat16> {
    static inline std::complex<at::Half> apply(at::BFloat16 src) {
      return static_cast<std::complex<at::Half>>(float{src});
    }
  };

  template <>
  struct static_cast_with_inter_type<std::complex<at::Half>, at::Half> {
    static inline std::complex<at::Half> apply(at::Half src) {
      return static_cast<std::complex<at::Half>>(float{src});
    }
  };

  template <>
  struct static_cast_with_inter_type<
      std::complex<at::Half>,
      std::complex<double>> {
    static inline std::complex<at::Half> apply(std::complex<double> src) {
      return static_cast<std::complex<at::Half>>(static_cast<std::complex<float>>(src));
    }
  };

  // Fetch a value with dynamic type src_type from ptr, and cast it to static type dest_t.
  #define FETCH_AND_CAST_CASE(type, scalartype) \
    case ScalarType::scalartype:                \
      return static_cast_with_inter_type<dest_t, type>::apply(c10::load<type>(ptr));
  template<typename dest_t>
  __device__ inline dest_t fetch_and_cast(const ScalarType src_type, const void *ptr) {
    switch (src_type) {
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_QINT(FETCH_AND_CAST_CASE)
        default:
          ERROR_UNSUPPORTED_CAST
    }
    return dest_t(0); // just to avoid compiler warning
  }

  // Cast a value with static type src_t into dynamic dest_type, and store it to ptr.
  #define CAST_AND_STORE_CASE(type, scalartype)                             \
    case ScalarType::scalartype:                                            \
      *(type*)ptr = static_cast_with_inter_type<type, src_t>::apply(value); \
      return;
  template<typename src_t>
  __device__ inline void cast_and_store(const ScalarType dest_type, void *ptr, src_t value) {
  switch (dest_type) {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_QINT(CAST_AND_STORE_CASE)
      default:;
  }
  ERROR_UNSUPPORTED_CAST
  }

  template <int N>
  struct LoadWithCast {
    using array_t = Array<ScalarType, N==0? 1 : N>;
    using size_array_t = Array<uint32_t, N==0? 1: N>;

    array_t dtypes;
    size_array_t element_sizes;
    template <typename scalar_t>
    __device__ scalar_t load(char* base_ptr, uint32_t offset, int arg) {
        void* ptr = base_ptr + element_sizes[arg] * offset;
        return fetch_and_cast<scalar_t>(dtypes[arg], ptr);
    }
  };

  template <int N = 1>
  struct StoreWithCast {
    using array_t = Array<ScalarType, N==0? 1 : N>;
    using size_array_t = Array<uint32_t, N==0? 1: N>;

    array_t dtypes;
    size_array_t element_sizes;

    template<typename scalar_t>
    __device__ void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
        void *ptr = base_ptr + element_sizes[arg] * offset;
        cast_and_store<scalar_t>(dtypes[arg], ptr, value);
    }
  };

)ESCAPE";

const std::string no_dynamic_cast_support_literal = R"ESCAPE(

  struct LoadWithoutCast {
  template <typename scalar_t>
  __device__ scalar_t load(char* base_ptr, uint32_t offset, int arg=0) {
    return c10::load(reinterpret_cast<scalar_t*>(base_ptr) + offset);
  }
  };

  struct StoreWithoutCast {
  template<typename scalar_t>
  __device__ void store(scalar_t value, char *base_ptr, uint32_t offset, int arg=0) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
  };

)ESCAPE";

const std::string offset_calc_template = R"ESCAPE(
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
      for (int dim = 0; dim < MAX_DIMS; ++dim) {
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
    IntDivider sizes_[MAX_DIMS];
    // NOTE: this approach will not support nInputs == 0
    ${index_type} strides_[MAX_DIMS][NARGS];
  };


)ESCAPE";

const std::string jit_code_template = R"ESCAPE(

  ${load_support}
  ${dynamic_casting_string}


  ${functor}

  // TODO: setup grid-stride loop
  extern "C" __global__
  void ${name}_kernel(
      const int numel,
      Array<char*, ${nInputs}+${nOutputs}> data, //[${nInputs}+${nOutputs}],
      ${offset_calculator}<${nInputs}> input_calculator,
      ${offset_calculator}<${nOutputs}> output_calculator,
      ${loader} l,
      ${storer} s,
      ${compute_type} scalar_val${extra_params}) {
    ${declare_load_arrays}
    ${declare_store_arrays}

    int idx = blockIdx.x;

    int remaining = numel - block_work_size * idx;
    int thread_idx = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < thread_work_size; j++){
        if (thread_idx >= remaining) {
            break;
        }

        int linear_idx = thread_idx + block_work_size * idx;
        auto input_offsets = input_calculator.get(linear_idx);
        ${load_inputs}
        // printf(
        //    "thread %d a %f offsets %d\n", threadIdx.x, arg0[j], input_offsets[0]);
        thread_idx += num_threads;
    }

    #pragma unroll
    for (int j = 0; j < thread_work_size; j++) {
      if ((threadIdx.x  + j*num_threads) < remaining) {
        ${call_functor}
      }
    }

    thread_idx = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < thread_work_size; j++){
        if (thread_idx >= remaining) {
            break;
        }
        //TODO maybe think about unifying offset calculators and reuse
        //offsets computed in the load loop
        int linear_idx = thread_idx + block_work_size * idx;
        auto output_offsets = output_calculator.get(linear_idx);
        //printf("output thread %d offset %d\n", threadIdx.x, output_offsets[0]);
        ${store_outputs}
        thread_idx += num_threads;
    }
  }
)ESCAPE";

const std::string jit_vectorized_code_template = R"ESCAPE(

  ${load_support}

  template <typename scalar_t>
  __device__ __inline__ scalar_t load(char* base_ptr, uint32_t offset) {
      return c10::load(reinterpret_cast<scalar_t*>(base_ptr) + offset);
  }

  template<typename scalar_t>
  __device__ __inline__ void store(scalar_t value, char *base_ptr, uint32_t offset) {
      *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }

  // aligned vector generates vectorized load/store on CUDA
  template<typename scalar_t, int vec_size>
  struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
    scalar_t val[vec_size];
  };

  template <int vec_size, typename scalar_t>
  __device__ aligned_vector<scalar_t, vec_size> load_vector(const scalar_t *base_ptr, uint32_t offset) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    auto *from = reinterpret_cast<const vec_t *>(base_ptr);
    return from[offset];
  }

  template <int vec_size>
  __device__ aligned_vector<bool, vec_size> load_vector(const bool *base_ptr, uint32_t offset) {
    // See NOTE [Loading boolean values]
    auto tmp = load_vector<vec_size>(reinterpret_cast<const uint8_t*>(base_ptr), offset);
    aligned_vector<bool, vec_size> ret;
    for (int i = 0; i < vec_size; ++i) {
      ret.val[i] = bool(tmp.val[i]);
    }
    return ret;
  }

  ${functor}

  // TODO: setup grid-stride loop

  extern "C" __global__
  void ${name}_vectorized${vec_size}_kernel(
      const int N,
      Array<char*, ${nInputs}+${nOutputs}> data,
      ${compute_type} scalar_val${extra_params}) //[${nInputs}+${nOutputs}],
      {
      constexpr int vec_size = ${vec_size};
      using scalar_t = ${scalar_type};
      int remaining = N - block_work_size * blockIdx.x;
      int thread_idx = threadIdx.x;
      int idx = blockIdx.x;
      ${declare_load_arrays}
      ${declare_store_arrays}

      if (remaining < block_work_size) {
        #pragma unroll
        for (int j = 0; j < thread_work_size; j++){
          if (thread_idx >= remaining) {
            break;
          }
          int linear_idx = thread_idx + block_work_size * idx;
          ${load_unrolled_inputs}
          thread_idx += num_threads;
        }
        #pragma unroll
        for (int j = 0; j < thread_work_size; j++) {
          if ((threadIdx.x  + j*num_threads) < remaining) {
            ${call_functor}
          }
        }
        thread_idx = threadIdx.x;
        #pragma unroll
        for (int j = 0; j < thread_work_size; j++) {
          if (thread_idx >= remaining) {
              break;
          }
          int linear_idx = thread_idx + block_work_size * idx;
          ${store_unrolled_outputs}
          thread_idx += num_threads;
        }
      } else {
        static constexpr int loop_size = thread_work_size / vec_size;
  //actual loading
        ${vector_inputs}
        #pragma unroll
        for (int i = 0; i<loop_size; i++){
          ${load_vectorized_inputs}
          thread_idx += num_threads;
        }

        #pragma unroll
        for (int j = 0; j < thread_work_size; j++) {
          ${call_functor}
        }

        using vec_t_output = aligned_vector<${result_type}, vec_size>;
        ${vector_outputs}
        int thread_idx = threadIdx.x;
        #pragma unroll
        for (int i = 0; i<loop_size; i++){
          vec_t_output v;
          ${store_vectorized_outputs}
          thread_idx += num_threads;
        }
      }
  }
)ESCAPE";

static void replace_all(std::string& s, const std::string& to_replace, const std::string& replace_with) {
  std::ostringstream oss;
  std::size_t pos = 0;
  std::size_t prev_pos = pos;

  while (true) {
    prev_pos = pos;
    pos = s.find(to_replace, pos);
    if (pos == std::string::npos)
      break;
    oss << s.substr(prev_pos, pos - prev_pos);
    oss << replace_with;
    pos += to_replace.size();
  }

  oss << s.substr(prev_pos);
  s = oss.str();
}

// hipify replaces certain device math functions, e.g., std::max -> ::max
// See torch/utils/hipify/cuda_to_hip_mappings.py.
// Replace them back. Search for " ::<name>" to avoid duplicate replacements.
static std::string unhipify_math_functions(const std::string &original) {
  static std::vector<std::pair<std::string,std::string>> mappings = {
    {" std::max", " ::max"},
    {" std::min", " ::min"},
    {" std::ceil", " ::ceil"},
    {" std::floor", " ::floor"},
    {" std::exp", " ::exp"},
    {" std::log", " ::log"},
    {" std::pow", " ::pow"},
    {" std::fabs", " ::fabs"},
    {" std::fmod", " ::fmod"},
    {" std::remainder", " ::remainder"},
    {" std::frexp", " ::frexp"}
  };
  std::string ret = original;
  for (const auto& mapping : mappings) {
    replace_all(ret, mapping.second, mapping.first);
  }
  return ret;
}

// The following is copied from fused_kernel.cpp
// TODO: refactor codegenOutputQuery into its own file
//   that can be included by both files
// See NOTE [ USE OF NVRTC AND DRIVER API ]
const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

// query codegen output arch and target
// TODO refactor so this function is usable both from jit and from aten
void codegenOutputQuery(
    const cudaDeviceProp* const prop,
    int& cuda_major,
    int& cuda_minor,
    int& nvrtc_major,
    int& nvrtc_minor,
    bool& compile_to_sass) {
#ifdef USE_ROCM
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));
  cuda_major = prop->major;
  cuda_minor = prop->minor;
  compile_to_sass = false;
#else
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));
  TORCH_CHECK(
      nvrtc_major >= 6, "NVRTC versions less than 6 are not supported. Is: ", nvrtc_major);

  // Version supported by device
  // Usually any lower version works too but is less efficient
  using CUDAVersion = std::pair<int, int>;
  const CUDAVersion nvrtc_version{nvrtc_major, nvrtc_minor};
  const CUDAVersion dev_version{prop->major, prop->minor};
  // Maximum version supported by the driver, cap dev_version to this
  CUDAVersion max_dev_version;
  if (nvrtc_major <= 7) { // 7 supports 2-5.x
    max_dev_version = CUDAVersion(5, 0);
  } else if (nvrtc_major <= 8) { // 8 supports 2-6.x
    max_dev_version = CUDAVersion(6, 0);
  } else if (nvrtc_major <= 9) { // 9 supports 3-7.2
    max_dev_version = CUDAVersion(7, 2);
  } else if (nvrtc_major <= 10) { // 10 supports 3-7.5
    max_dev_version = CUDAVersion(7, 5);
  } else if (nvrtc_version == CUDAVersion(11, 0)) { // 11.0 supports 3-8.0
    max_dev_version = CUDAVersion(8, 0);
  } else if (nvrtc_major == 11 && nvrtc_minor < 8) {
    max_dev_version = CUDAVersion(8, 6);
  } else {
    // If the driver version is unknown (i.e. newer than this code)
    // assume the driver supports this device
    max_dev_version = dev_version;
  }

  if (dev_version > max_dev_version) {
    cuda_major = max_dev_version.first;
    cuda_minor = max_dev_version.second;
    // if we are clamping major/minor, sass is not compatible
    compile_to_sass = false;
  } else {
    cuda_major = dev_version.first;
    cuda_minor = dev_version.second;
    compile_to_sass = true;
  }

  #if defined(CUDA_VERSION) && CUDA_VERSION < 11010
    // compile to sass is not allowed prior to CUDA 11.1
    compile_to_sass = false;
  #endif
#endif
}

// TODO: another copy paste from jit, refactor so it's usable from both
// TODO: try making the CUcontext thread local to see if that improves performance - why is this slow?
void initializeCudaContext() {
  // lazily construct context if non-existing yet;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  CUcontext pctx = nullptr;
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::getFreeMutex()));
    cudaFree(nullptr);
  }
}

#ifdef USE_ROCM
int calc_io_size(
    const int nInputs,
    const int nOutputs,
    const c10::ScalarType& inputs_type,
    const c10::ScalarType& result_type) {
    if (nInputs > 0 && nOutputs > 0) {
        return std::min(c10::elementSize(inputs_type), c10::elementSize(result_type));
    }

    if (nInputs > 0) {
        return c10::elementSize(inputs_type);
    }

    if (nOutputs > 0) {
        return c10::elementSize(result_type);
    }

    return 0;
}
#endif

int calc_thread_work_size(
    const int nInputs,
    const int nOutputs,
    const c10::ScalarType& inputs_type,
    const c10::ScalarType& result_type) {
#ifdef USE_ROCM
    auto io_size = at::cuda::jit::calc_io_size(nInputs, nOutputs, inputs_type, result_type);
    TORCH_INTERNAL_ASSERT(io_size > 0);
    if (io_size == 1) {
        return 16;
    } else if (io_size < 4) {
        return 8;
    } else {
        return 4;
    }
    return io_size;
#else
    return JIT_THREAD_WORK_SIZE;
#endif
}

std::string generate_code(
    const KernelDescriptor &desc,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    int thread_work_size,
    bool vectorized,
    int vec_size,
    bool return_by_ref) {
  c10::SmallVector<std::string> extra_args_typenames(desc.extra_args_types.size());
  for (auto i : c10::irange(extra_args_typenames.size())) {
    extra_args_typenames[i] = typeName(desc.extra_args_types[i]);
  }

  return generate_code(
      desc.nInputs,
      desc.nOutputs,
      desc.f,
      desc.name,
      typeName(desc.f_inputs_type),
      typeName(toOpMathType(desc.f_inputs_type)),
      typeName(desc.result_type),
      contiguous,
      dynamic_casting,
      scalar_pos,
      extra_args_typenames,
      thread_work_size,
      vectorized,
      vec_size,
      return_by_ref);
}

std::string generate_code(
    int nInputs,
    int nOutputs,
    const std::string& func_,
    const std::string& name,
    const std::string& f_inputs_type,
    const std::string& compute_type,
    const std::string& result_type,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    c10::SmallVector<std::string>& extra_args_typenames,
    int thread_work_size,
    bool vectorized,
    int vec_size,
    bool return_by_ref) {
  std::string func = func_;
  at::jit::TemplateEnv env;

  env.s("index_type", "unsigned int");
  env.s("nInputs", std::to_string(nInputs));
  env.s("nOutputs", std::to_string(nOutputs));
  env.s("scalar_type", f_inputs_type);
  env.s("compute_type", compute_type);
  env.s("functor", func);
  env.s("name", name);
  env.s("cmath_string", get_cmath_string());
  env.s("thread_work_size", std::to_string(thread_work_size));

  // Generate `extra_params` for function signature
  // and `extra_args` for computation call if
  // extra arguments to capture runtime state are passed.
  // (look at polygamma for example).
  std::string extra_params = "";
  std::string extra_args = "";
  for (size_t i = 0; i < extra_args_typenames.size(); i++) {
    auto type = std::string(extra_args_typenames[i]);
    auto name = "extra_arg_" + std::to_string(i);
    extra_params += "," + type + " " + name;
    extra_args += ", " + name;
  }
  env.s("extra_params", extra_params);
  env.s("extra_args", extra_args);

  std::stringstream declare_load_arrays;
  for (int i = 0; i < nInputs; i++) {
    // TODO these arrays are potentially of the different types, use function
    // traits to determine the types
    declare_load_arrays << f_inputs_type << " arg" << std::to_string(i)
                        << "[" << std::to_string(thread_work_size) << "];\n";
  }
  env.s("declare_load_arrays", declare_load_arrays.str());

  std::stringstream declare_store_arrays;
  for (int i = 0; i < nOutputs; i++) {
    declare_store_arrays << result_type << " out" << std::to_string(i)
                        << "[" << std::to_string(thread_work_size) << "];\n";
  }
  env.s("declare_store_arrays", declare_store_arrays.str());

  std::stringstream functor_args;
  if (scalar_pos == BinaryFuncVariant::NoScalar) {
    for (int i = 0; i < nInputs - 1; i++) {
      functor_args << "arg" << std::to_string(i) << "[j], ";
    }
    functor_args << "arg" << std::to_string(nInputs - 1) << "[j]";
  } else if (scalar_pos == BinaryFuncVariant::LhsScalar) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(nInputs == 1);
    functor_args << "scalar_val, arg0[j]";
  } else { //RhsScalar
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(nInputs == 1);
    functor_args << "arg0[j], scalar_val";
  }
  env.s("args", functor_args.str());

  std::string call_functor_template;
  if (return_by_ref) {  // return one or more outputs by reference
    bool need_temp_out = (compute_type != result_type);
    std::stringstream functor_outs;
    if (need_temp_out) {
      for (int i = 0; i < nOutputs - 1; i++) {
        functor_outs << "temp_out" << std::to_string(i) << ", ";
      }
      functor_outs << "temp_out" << std::to_string(nOutputs - 1);
    } else {
      for (int i = 0; i < nOutputs - 1; i++) {
        functor_outs << "out" << std::to_string(i) << "[j], ";
      }
      functor_outs << "out" << std::to_string(nOutputs - 1) << "[j]";
    }
    env.s("functor_outs", functor_outs.str());

    if (need_temp_out) {
      call_functor_template += "${compute_type} ${functor_outs};\n";
    }

    call_functor_template += "${name}<${compute_type}>(${args} ${extra_args}, ${functor_outs});\n";

    if (need_temp_out) {
      for (int i = 0; i < nOutputs; i++) {
        auto i_string = std::to_string(i);
        call_functor_template += "out" +i_string + "[j] = temp_out" + i_string + ";\n";
      }
    }

  } else {  // return by value for single output functor
    call_functor_template = "out0[j] = ${name}<${compute_type}>(${args} ${extra_args});";
  }
  env.s("call_functor", at::jit::CodeTemplate(call_functor_template).format(env));

  if (f_inputs_type == "at::Half" || result_type == "at::Half" ||
      f_inputs_type == "std::complex<at::Half>" ||
      result_type == "std::complex<at::Half>" || dynamic_casting) {
    // complex<Half> depends on complex<T> and Half dtypes.
    env.s("half_string", jiterator_half_support_literal);
  } else {
    env.s("half_string", "");
  }
  if (f_inputs_type == "at::BFloat16" || result_type == "at::BFloat16" || dynamic_casting) {
    env.s("bfloat16_string", jiterator_bfloat16_support_literal);
  } else {
    env.s("bfloat16_string", "");
  }
  // the definition of complex math functions is only needed when the compute type is complex
  // but the definition of std::complex is needed for dynamic casting even if the compute type is not complex
  if (f_inputs_type == "std::complex<float>" || result_type == "std::complex<float>" ||
      f_inputs_type == "std::complex<double>" || result_type == "std::complex<double>" ||
      f_inputs_type == "std::complex<at::Half>" || result_type == "std::complex<at::Half>") {
    // complex<Half> depends on complex<T> and Half dtypes.
    env.s("traits_string", get_traits_string_but_hiprtc_safe());
    env.s("complex_body_string", get_complex_body_string());
    env.s("complex_math_string", get_complex_math_string());
#ifdef USE_ROCM
    // unhipify math functions, but only if std::complex is used.
    func = unhipify_math_functions(func);
    env.s("functor", func);
#endif
  } else if (dynamic_casting) {
    env.s("traits_string", get_traits_string_but_hiprtc_safe());
    env.s("complex_body_string", get_complex_body_string());
    env.s("complex_math_string", "");
  } else {
    env.s("traits_string", "");
    env.s("complex_body_string", "");
    env.s("complex_math_string", "");
  }
  if (f_inputs_type == "std::complex<at::Half>" ||
      result_type == "std::complex<at::Half>" || dynamic_casting) {
    // dynamic_casting requires the definition of all types
    // include complex<at::Half>
    // Look at the definition of `StoreWithCast` and `LoadWithCast`.
    env.s("complex_half_body_string", get_complex_half_body_string());
  } else {
    env.s("complex_half_body_string", "");
  }

  env.s("load_support", load_support_literal);

  if (!vectorized) {
    if (!dynamic_casting) {
      env.s("loader", "LoadWithoutCast");
      env.s("storer", "StoreWithoutCast");
      env.s("dynamic_casting_string", no_dynamic_cast_support_literal);
    } else {
      env.s("loader", std::string("LoadWithCast<" + std::to_string(nInputs) + ">"));
      env.s("storer", std::string("StoreWithCast<" + std::to_string(nOutputs) + ">"));
      env.s("dynamic_casting_string", dynamic_cast_support_literal);
    }

    if (contiguous) {
      env.s("offset_calculator", "TrivialOffsetCalculator");
    } else {
      env.s("offset_calculator", "OffsetCalculator");
    }

    std::stringstream load_inputs;
    for (int i = 0; i < nInputs; i++) {
      auto i_string = std::to_string(i);
      load_inputs << "arg" << i_string << "[j] = l.load<" << f_inputs_type
                  << ">(data[" << std::to_string(i + nOutputs)
                  << "], input_offsets[" << i_string << "], " << i_string
                  << ");\n";
    }
    env.s("load_inputs", load_inputs.str());

    std::stringstream store_outputs;
    for (int i = 0; i < nOutputs; i++) {
      auto i_string = std::to_string(i);
      store_outputs << "s.store<" << result_type
                    << ">(out" << i_string << "[j], data[" << i_string
                    << "], output_offsets[" << i_string << "], " << i_string
                    << ");\n";
    }
    env.s("store_outputs", store_outputs.str());

    static auto cuda_template = at::jit::CodeTemplate(
      jit_preamble + jit_common_types + offset_calc_template + jit_code_template + jit_epilogue);
    const auto code = cuda_template.format(env);
    return code;
  }

  // vectorized case
  env.s("vec_size", std::to_string(vec_size));
  env.s("result_type", result_type);

  std::stringstream vector_inputs;
  for (const auto i : c10::irange(nInputs)){
    auto i_string = std::to_string(i);
    vector_inputs << "auto * input" << i_string <<
        " = reinterpret_cast<const scalar_t*>(data[" << i_string << "+" << nOutputs << "])" <<
        " + block_work_size * idx;\n";
  }
  env.s("vector_inputs", vector_inputs.str());

  std::stringstream vector_outputs;
  for (const auto i : c10::irange(nOutputs)){
    auto i_string = std::to_string(i);
    vector_outputs << "vec_t_output* to_" << i_string <<
    " = reinterpret_cast<vec_t_output*>(data[" << i_string << "])" <<
    " + block_work_size / vec_size * idx;\n";
  }
  env.s("vector_outputs", vector_outputs.str());

  std::stringstream load_vectorized_inputs;
  for (const auto i : c10::irange(nInputs)) {
    auto i_string = std::to_string(i);
    load_vectorized_inputs << "const auto vec" << i_string << " = load_vector<vec_size>("
                           << "input" << i_string << ", thread_idx);\n";
    load_vectorized_inputs << "#pragma unroll\n";
    load_vectorized_inputs << "for (int j=0; j < vec_size; j++){\n";
    load_vectorized_inputs << "  arg" << i_string << "[vec_size * i + j] = vec" << i_string << ".val[j];\n";
    load_vectorized_inputs << "}\n";
  }
  env.s("load_vectorized_inputs", load_vectorized_inputs.str());

  std::stringstream store_vectorized_outputs;
  for (const auto i : c10::irange(nOutputs)) {
    auto i_string = std::to_string(i);
    store_vectorized_outputs << "#pragma unroll\n";
    store_vectorized_outputs << "for (int j=0; j<vec_size; j++){\n";
    store_vectorized_outputs <<   "v.val[j] = out" << i_string << "[vec_size * i + j];\n";
    store_vectorized_outputs << "}\n";
    store_vectorized_outputs << "to_"<< i_string << "[thread_idx] = v;\n";
  }
  env.s("store_vectorized_outputs", store_vectorized_outputs.str());

  std::stringstream load_unrolled_inputs;
  for (const auto i: c10::irange(nInputs)){
    auto i_string = std::to_string(i);
    load_unrolled_inputs << "arg" << i_string << "[j] = load<" << f_inputs_type
      << ">(data[" << std::to_string(i + nOutputs) << "], linear_idx);\n";
  }
  env.s("load_unrolled_inputs", load_unrolled_inputs.str());

  std::stringstream store_unrolled_outputs;
  for (const auto i : c10::irange(nOutputs)) {
    auto i_string = std::to_string(i);
    store_unrolled_outputs << "store<" << result_type << ">(out" << i_string
      << "[j], data[" << i_string << "], linear_idx);\n";
  }
  env.s("store_unrolled_outputs", store_unrolled_outputs.str());

  static auto cuda_template = at::jit::CodeTemplate(
    jit_preamble + jit_common_types + jit_vectorized_code_template + jit_epilogue);
  const auto code = cuda_template.format(env);
  return code;
}

// Creates directories recursively
bool _r_mkdir(const std::string& dir) {
  // Check if current dir exists
  const char* p_dir = dir.c_str();
  const bool dir_exists = (access(p_dir, F_OK) == 0);
  if (dir_exists) {
    return true;
  }

  // Try to create current directory
#ifdef _WIN32
  int ret = _mkdir(dir.c_str());
#else
  int ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
  // Success
  if (ret == 0) {
    return true;
  }

  // Find folder separator and check if we are at the top
  auto  pos = dir.find_last_of("/\\");
  if (pos == std::string::npos) {
    return false;
  }

  // Try to create parent directory
  if (!(_r_mkdir(dir.substr(0, pos)))) {
    return false;
  }

  // Try to create complete path again
#ifdef _WIN32
  ret = _mkdir(dir.c_str());
#else
  ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
  return ret == 0;
}

// Creates directories recursively assuming that base exists
bool r_mkdir_with_base(std::string& base, std::string& dir){
  const char* p_base = base.c_str();
  const bool base_exists = (access(p_base, F_OK) == 0);
  if (!base_exists) {
    return false;
  }

  // remove trailing '/' or '\\'
  if ((base[base.size()-1]=='/') || base[base.size()-1]=='\\') {
    base.pop_back();
  }
  if ((dir[dir.size()-1]=='/') || dir[dir.size()-1]=='\\') {
    dir.pop_back();
  }

  return _r_mkdir(base+dir);

}

std::string load_code_template(const std::string& path) {
  std::ifstream ifs{path};
  std::string s{
    std::istreambuf_iterator<char>(ifs),
    std::istreambuf_iterator<char>()};
  return s;
}

std::string generate_reduction_code(
    const KernelDescriptor &desc,
    int vt0,
    bool contiguous,
    bool vectorized,
    int vec_size,
    int max_threads_codegen) {
  TORCH_INTERNAL_ASSERT(desc.nInputs == 1);
  TORCH_INTERNAL_ASSERT(desc.extra_args_types.size() == 0);

  return generate_reduction_code(
      desc.nOutputs,
      desc.f,
      desc.name,
      vt0,
      typeName(desc.f_inputs_type),
      typeName(toOpMathType(desc.f_inputs_type)),
      typeName(desc.result_type),
      contiguous,
      vectorized,
      vec_size,
      max_threads_codegen
    );
}

std::string generate_reduction_code(
    int nOutputs,
    const std::string& func_,
    const std::string& name,
    const int vt0,
    const std::string& f_inputs_type,
    const std::string& reduction_accum_type,
    const std::string& result_type,
    bool contiguous,
    bool vectorized,
    int vec_size,
    int max_threads_codegen) {
      std::string func = func_;
      at::jit::TemplateEnv env;
      constexpr int thread_work_size = JIT_THREAD_WORK_SIZE;
      env.s("index_type", "unsigned int");
      env.s("scalar_type", f_inputs_type);
      env.s("result_type", result_type);
      env.s("reduction_accum_type", reduction_accum_type);
      env.s("vt0", std::to_string(vt0));
      env.s("name", name);
      env.s("max_threads_lb", std::to_string(max_threads_codegen));
      env.s("thread_work_size", std::to_string(thread_work_size));
      // reductions don't support dynamic casting, so the only way to get nonstandard types
      // is through input
      if (f_inputs_type == "at::Half" || f_inputs_type == "std::complex<at::Half>") {
        // complex<Half> depends on complex<T> and Half dtypes.
        env.s("half_string", jiterator_half_support_literal);
      } else {
        env.s("half_string", "");
      }
      if (f_inputs_type == "at::BFloat16") {
        env.s("bfloat16_string", jiterator_bfloat16_support_literal);
      } else {
        env.s("bfloat16_string", "");
      }
      if (f_inputs_type == "std::complex<float>" ||
          f_inputs_type == "std::complex<double>" ||
          f_inputs_type == "std::complex<at::Half>" ) {
        // complex<Half> depends on complex<T> and Half dtypes.
        env.s("traits_string", get_traits_string_but_hiprtc_safe());
        env.s("complex_body_string", get_complex_body_string());
        env.s("complex_math_string", get_complex_math_string());
        env.s("complex", std::to_string(1));
#ifdef USE_ROCM
        // unhipify math functions, but only if std::complex is used.
        func = unhipify_math_functions(func);
#endif
      } else {
        env.s("traits_string", "");
        env.s("complex_body_string", "");
        env.s("complex_math_string", "");
        env.s("complex", std::to_string(0));
      }
      if (f_inputs_type == "std::complex<at::Half>") {
        env.s("complex_half_body_string", get_complex_half_body_string());
      } else {
        env.s("complex_half_body_string", "");
      }
      env.s("cmath_string", get_cmath_string());
      env.s("functor", func);
      env.s("output_vec_size", std::to_string(vec_size));
      static auto cuda_template = at::jit::CodeTemplate(
        jit_preamble + jit_common_types + offset_calc_template + get_reduction_template() + jit_epilogue);
      const auto code = cuda_template.format(env);
      return code;
}

// Acquires (possibly creating) the kernel cache directory
std::optional<std::string> get_cache_dir() {
  // If the environment variable USE_TORCH_KERNEL_CACHE is set to "0" then no persistent cache is used
  const char* uptkc = std::getenv("USE_PYTORCH_KERNEL_CACHE");
  const bool use_kernel_cache = (uptkc == nullptr) ? true : std::strcmp(uptkc, "0");

  if (!use_kernel_cache) {
    return {};
  }

  // Cache path comes from PYTORCH_KERNEL_CACHE_PATH, then TEMP (Windows) or XDG_CACHE_HOME (Linux), then HOME environment variables
  std::string cache_dir;
  char* ptkcp = std::getenv("PYTORCH_KERNEL_CACHE_PATH");
  // Create kernel_cache_dir if needed as we do not want to create the base directory passed by the user
  std::string kernels_cache_dir = "";
  if (ptkcp != nullptr) {
    cache_dir = std::string(ptkcp);
  } else {
#ifdef _WIN32
    ptkcp = std::getenv("TEMP");
#else
    // USES XDG_CACHE_HOME if it's set
    ptkcp = std::getenv("XDG_CACHE_HOME");
#endif
    if (ptkcp != nullptr) {
      kernels_cache_dir = "/torch/kernels";
      cache_dir = std::string(ptkcp) + kernels_cache_dir;
    } else {
      // Falls back to HOME/.cache
      ptkcp = std::getenv("HOME");
      if (ptkcp == nullptr) {
        TORCH_WARN_ONCE("No PYTORCH_KERNEL_CACHE_PATH or HOME environment variable set!",
                        " This disables kernel caching.");
        return {};
      } else {
        kernels_cache_dir = "/.cache/torch/kernels";
        cache_dir = std::string(ptkcp) + kernels_cache_dir;
      }
    }
  }

  // Creates the cache directory if it does not exist
  const char* p_cache_dir = cache_dir.c_str();
  const bool cache_dir_exists = (access(p_cache_dir, F_OK) == 0);
  if (!cache_dir_exists) {
    std::string s_ptkcp = std::string(ptkcp);
    if (!r_mkdir_with_base(s_ptkcp, kernels_cache_dir)) {
      TORCH_WARN_ONCE("Specified kernel cache directory could not be created! This disables kernel caching.",
                      " Specified directory is ", cache_dir, ".",
                      " This warning will appear only once per process.");
      return {};
    }
  }

  // Checks that the cache directory is readable and writable
  const bool cache_dir_readable = (access(p_cache_dir, R_OK) == 0);
  if (!cache_dir_readable) {
    TORCH_WARN_ONCE("Specified kernel cache directory is not readable! This disables kernel caching.",
                    " Specified directory is ", cache_dir, ".",
                    " This warning will appear only once per process.");
    return {};
  }

  const bool cache_dir_writable = (access(p_cache_dir, W_OK) == 0);
  if (!cache_dir_writable) {
    TORCH_WARN_ONCE("Specified kernel cache directory is not writable! This disables kernel caching.",
                    " Specified directory is ", cache_dir, ".",
                    " This warning will appear only once per process.");
    return {};
  }

  return cache_dir;
}

// Compiles the kernel, or acquires if from the cache if caching
NvrtcFunction jit_pwise_function(
    const std::string& code,
    const std::string& kernel_name) {
  initializeCudaContext();
  // Acquires CUDA and nvrtc versions and whether we're compiling to ptx or SASS
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int cuda_major = 0, cuda_minor = 0, nvrtc_major = 0, nvrtc_minor = 0;
  bool compile_to_sass = false;
  at::cuda::jit::codegenOutputQuery(
    prop, cuda_major, cuda_minor, nvrtc_major, nvrtc_minor, compile_to_sass);

  // Objects used whether loading from the cache or jit compiling
  const auto& nvrtc = at::globalContext().getNVRTC();
  NvrtcFunction compiled_kernel_;
  std::string name = kernel_name + "_kernel";

  static const std::optional<std::string> cache_dir = get_cache_dir();

  std::string file_path;
  if (cache_dir.has_value()) {
    // Attemps to read from the cache.
    // Cubin name is <kernel name>_arch<major>.<minor>_nvrtc<major>.<minor>_<ptx or sass>_<program length>_<string hash>
    // Note that the SHA1 hash used in the file name is NOT the SHA1 hash of the file's contents,
    //   because we hash on the CUDA code, but we save the compiled ptx or sass

    // Acquires SHA1 hash
    c10::sha1 sha1_hash{code};
    const auto hash_code = sha1_hash.str();

    // Constructs file path by appending constructed cubin name to cache path
    std::stringstream ss;
    ss << *cache_dir << "/";
    ss << kernel_name;
#ifdef USE_ROCM
    ss << "_arch" << prop->gcnArchName;
#else
    ss << "_arch" << cuda_major << "." << cuda_minor;
#endif
    ss << "_nvrtc" << nvrtc_major << "." << nvrtc_minor;
    ss << (compile_to_sass ? "_sass" : "_ptx");
    ss << "_" << code.length();
    ss << "_" << hash_code;
    file_path = ss.str();

    std::ifstream readin{file_path, std::ios::in | std::ifstream::binary};
    if (readin.fail()) {
      // NOTE: this does not warn because the file might not exist
      // TODO: consider if this should explicitly check for the file's existence or not to throw
      //   an informative warning
      readin.close();
    } else {
      // TODO: try passing the "mapped" file directly to cuModuleLoadCall instead of using an intermediate buffer
      std::vector<char> buffer(std::istreambuf_iterator<char>(readin), {});
      AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&(compiled_kernel_.module), buffer.data()));
      AT_CUDA_DRIVER_CHECK(
        nvrtc.cuModuleGetFunction(&(compiled_kernel_.function), compiled_kernel_.module, name.c_str()));
      readin.close();
      return compiled_kernel_;
    }
  }

  // Just-in-time compiles the program

  // Creates the NVRTC program
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));

#ifdef USE_ROCM
  std::vector<const char*> args = {"--std=c++17"};
#else
  // Constructs nvrtc build arguments
  // CUDA 11.1 allows going directly to SASS (sm_) instead of PTX (compute_)
  // which gives better backwards compatibility to work on older driver,
  // (since older driver doesn't necessarily recognize PTX emitted by new
  // toolkit);
  // Meanwhile, for forward compatibility (future device with
  // `unsupported_arch==True`), since SASS are not necessarily compatible,
  // we fallback to PTX instead.
  const std::string compute = std::string("--gpu-architecture=") +
      (compile_to_sass ? "sm_" : "compute_") + std::to_string(cuda_major) +
      std::to_string(cuda_minor);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<const char*> args = {
      "--std=c++17", compute.c_str(), "-default-device"};
#endif

  #ifndef NDEBUG
    // Add line info to generated kernels
    args.push_back("-lineinfo");
  #else
    // Avoid excessive register usage from assertion
    args.push_back("-DNDEBUG");
  #endif

  const auto compilation_result =
      nvrtc.nvrtcCompileProgram(program, args.size(), args.data());

  // Throws an error on compilation failure
  if (compilation_result != NVRTC_SUCCESS) {
    size_t logsize;
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLogSize(program, &logsize));
    std::string log(logsize, '\0');
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLog(program, &log[0]));
    throw std::runtime_error(code + log);
  }

  size_t ptx_size = 0;
  std::vector<char> ptx;
  #if defined(CUDA_VERSION) && CUDA_VERSION >= 11010
    // compile_to_sass determines whether we are generating SASS or PTX, hence
    // the different API.
    const auto getSize = compile_to_sass
        ? at::globalContext().getNVRTC().nvrtcGetCUBINSize
        : at::globalContext().getNVRTC().nvrtcGetPTXSize;
    const auto getFunc = compile_to_sass
        ? at::globalContext().getNVRTC().nvrtcGetCUBIN
        : at::globalContext().getNVRTC().nvrtcGetPTX;
  #else
    const auto getSize = at::globalContext().getNVRTC().nvrtcGetPTXSize;
    const auto getFunc = at::globalContext().getNVRTC().nvrtcGetPTX;
  #endif

  AT_CUDA_NVRTC_CHECK(getSize(program, &ptx_size));
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(getFunc(program, ptx.data()));

  AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&(compiled_kernel_.module), ptx.data()));

  AT_CUDA_DRIVER_CHECK(
      nvrtc.cuModuleGetFunction(&(compiled_kernel_.function), compiled_kernel_.module, name.c_str()));
  // TODO: use guards to avoid leaking
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcDestroyProgram(&program));

  if (cache_dir.has_value()) {
    // Writes the program to the cache if caching
    // NOTE: Actually writes to a per-process temporary file to avoid multi-process contention.
    //   The temporary file is then renamed to the actual file.
    //   If the actual file already exists then the rename may fail or replace the actual file,
    //     the behavior is implementation-specific.
    //   Files replaced through this process should remain extant if they are being read because
    //     of UNIX filesystem properties, but this behavior is unverified and may require
    //     additional review in the future.
    // TODO: In C++17 we should be able to use the filesystem header.
    const auto pid = getpid();
    std::stringstream tmp_file_path_ss;
    tmp_file_path_ss << file_path << "_tmp_" << pid;
    const std::string tmp_file_path = tmp_file_path_ss.str();
    std::ofstream cubin(tmp_file_path, std::ios::out | std::ofstream::binary);
    if (cubin.fail()) {
      TORCH_WARN_ONCE("Failed to write temporarily kernel cache file!",
                      " File path was ", tmp_file_path, ".",
                      " This warning will only appear once per process.");
    } else {
      std::copy(ptx.begin(), ptx.end(), std::ostreambuf_iterator<char>(cubin));
      if (std::rename(tmp_file_path.c_str(), file_path.c_str()) != 0) {
        // Removes tmp file if the rename failed
        std::remove(tmp_file_path.c_str());
      }
    }
    cubin.close();
  }

  return compiled_kernel_;
}

// TODO: may need/want to initialize CUDA context here (refactor into nvrtc call)
void launch_jitted_pwise_function(
    NvrtcFunction function,
    const void* args[],
    const dim3 nBlocks,
    const dim3 kBlockSize,
    const int smem) {
  initializeCudaContext();
  const auto& nvrtc = at::globalContext().getNVRTC();
  // Launches kernel on current stream
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_DRIVER_CHECK(nvrtc.cuLaunchKernel(
    function.function,
    nBlocks.x,
    nBlocks.y,
    nBlocks.z,
    kBlockSize.x,
    kBlockSize.y,
    kBlockSize.z,
    smem,
    stream,
    // NOLINTNEXTLINE(*const-cast*)
    const_cast<void**>(args),
    nullptr));
}

} // at::cuda::jit
