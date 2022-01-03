#include <sstream>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>

namespace at { namespace cuda { namespace jit {

const std::string jit_common_types = R"ESCAPE(
  typedef long long int int64_t;
  typedef unsigned int uint32_t;
  typedef signed char int8_t;
  typedef unsigned char uint8_t;  // NOTE: this MUST be "unsigned char"! "char" is equivalent to "signed char"
  typedef short int16_t;
  static_assert(sizeof(int64_t) == 8, "expected size does not match");
  static_assert(sizeof(uint32_t) == 4, "expected size does not match");
  static_assert(sizeof(int8_t) == 1, "expected size does not match");
  constexpr int num_threads = 128;
  constexpr int thread_work_size = 4; // TODO: make template substitution once we decide where those vars live
  constexpr int block_work_size = thread_work_size * num_threads;
  //TODO use _assert_fail, because assert is disabled in non-debug builds
  #define ERROR_UNSUPPORTED_CAST assert(false);


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
  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */        \
  _(c10::complex<float>, ComplexFloat) /* 9 */                          \
  _(c10::complex<double>, ComplexDouble) /* 10 */                         \
  _(bool, Bool) /* 11 */                                 \
  _(void, QInt8) /* 12 */                          \
  _(void, QUInt8) /* 13 */                        \
  _(void, QInt32) /* 14 */                        \
  _(at::BFloat16, BFloat16) /* 15 */                             \

  #define AT_FORALL_SCALAR_TYPES(_) \
  _(uint8_t, Byte)                \
  _(int8_t, Char)                 \
  _(int16_t, Short)               \
  _(int, Int)                     \
  _(int64_t, Long)                \
  _(float, Float)                 \
  _(at::Half, Half)               \
  _(at::BFloat16, BFloat16)       \
  _(double, Double)               \
  _(bool, Bool)                   \
  _(c10::complex<float>, ComplexFloat)   \
  _(c10::complex<double>, ComplexDouble)


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
  };

  ${half_string}
  ${bfloat16_string}
  ${complex_string}


)ESCAPE";

//we need to include half, bfloat16 and complex strings to all kernels with half arguments and to all kernels with type casting
//regardless of whether they have half arguments (because fetch_and_cast and cast_and_store loop over all types)
const std::string jiterator_half_support_literal = R"ESCAPE(
namespace at {
struct alignas(2) Half {
  unsigned short x;

  Half() = default;
  inline __host__ __device__ Half(float value){
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(x) : "f"(value));
  }
  inline __host__ __device__ operator float() const{
      float val;
      asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(x)); // do we need const cast here?
      //asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(x)));
      return val;
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
    float val;
    asm("{ mov.b32 %0, {0,%1};}\n" : "=f"(val) : "h"(x)); //do we need const cast here?
    return val;
  }

};
}
)ESCAPE";

//copy-pasted from util/complex.h
const std::string jiterator_complex_support_literal = R"ESCAPE(
//a very limited complex class, the only thing it currently allows is implicit conversion
//to complex, and complex -> real that is unused
namespace c10 {
  template<typename T>
  struct alignas(sizeof(T) * 2) complex {
    using value_type = T;

    T real_ = T(0);
    T imag_ = T(0);
    constexpr complex() = default;
    inline __host__ __device__ constexpr complex(const T& re, const T& im = T())
      : real_(re), imag_(im) {}

    //FIXME I didn't find how complex -> real conversion is done in eager
    //we are not going to use it, but it's needed for compilation
    inline __host__ __device__ operator T() const{
      return real_;
    }

  };
}
)ESCAPE";


const std::string jit_code_template = R"ESCAPE(

  // Fetch a value with dynamic type src_type from ptr, and cast it to static type dest_t.
  // For now, simplified version that does not handle complex and special casting to uint8
  #define FETCH_AND_CAST_CASE(type, scalartype) case ScalarType::scalartype: return static_cast<dest_t>(*(const type *)ptr);
  template<typename dest_t>
  __device__ inline dest_t fetch_and_cast(const ScalarType src_type, const void *ptr) {
    switch (src_type) {
        AT_FORALL_SCALAR_TYPES(FETCH_AND_CAST_CASE)
        default:
          ERROR_UNSUPPORTED_CAST
    }
    return dest_t(0); // just to avoid compiler warning
  }

  // Cast a value with static type src_t into dynamic dest_type, and store it to ptr.
  #define CAST_AND_STORE_CASE(type, scalartype) case ScalarType::scalartype: *(type *)ptr = static_cast<type>(value); return;
  template<typename src_t>
  __device__ inline void cast_and_store(const ScalarType dest_type, void *ptr, src_t value) {
  switch (dest_type) {
      AT_FORALL_SCALAR_TYPES(CAST_AND_STORE_CASE)
      default:;
  }
  ERROR_UNSUPPORTED_CAST
  }

  struct LoadWithoutCast {
  template <typename scalar_t>
  __device__ scalar_t load(char* base_ptr, uint32_t offset, int arg=0) {
      return *(reinterpret_cast<scalar_t*>(base_ptr) + offset);
  }
  };

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

  struct StoreWithoutCast {
  template<typename scalar_t>
  __device__ void store(scalar_t value, char *base_ptr, uint32_t offset) {
      *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
  };

  struct StoreWithCast {
  ScalarType dtype;
  uint32_t element_size;
  //StoreWithCast(at::ScalarType dtype): dtype(dtype), element_size(c10::elementSize(dtype)) {}
  template<typename scalar_t>
  __device__ void store(scalar_t value, char *base_ptr, uint32_t offset) {
      void *ptr = base_ptr + element_size * offset;
      cast_and_store<scalar_t>(dtype, ptr, value);
  }
  };

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

  ${functor}

  // TODO: setup grid-stride loop
  extern "C" __global__
  void ${name}_kernel(
      const int numel,
      Array<char*, ${nInputs}+1> data, //[${nInputs}+1],
      ${offset_calculator}<${nInputs}> input_calculator,
      ${offset_calculator}<1> output_calculator,
      ${loader} l,
      ${storer} s,
      ${compute_type} scalar_val) {
    ${declare_load_arrays}
    ${declare_store_arrays}

    int idx = blockIdx.x;

    int remaining = numel - block_work_size * idx;
    auto thread_idx = threadIdx.x;

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
        out[j] = ${name}<${compute_type}>(${args});
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
        //TODO handle multi-return functors
        ${store_outputs}
        thread_idx += num_threads;
    }
  }
)ESCAPE";

const std::string jit_vectorized_code_template = R"ESCAPE(

  template <typename scalar_t>
  __device__ __inline__ scalar_t load(char* base_ptr, uint32_t offset) {
      return *(reinterpret_cast<scalar_t*>(base_ptr) + offset);
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

  ${functor}

  // TODO: setup grid-stride loop

  extern "C" __global__
  void ${name}_vectorized${vec_size}_kernel(
      const int N,
      Array<char*, ${nInputs}+1> data,
      ${compute_type} scalar_val) //[${nInputs}+1],
      {
      constexpr int vec_size = ${vec_size};
      int remaining = N - block_work_size * blockIdx.x;
      auto thread_idx = threadIdx.x;
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
            out[j] = ${name}<${compute_type}>(${args});
          }
        }
        thread_idx = threadIdx.x;
        #pragma unroll
        for (int j = 0; j < thread_work_size; j++) {
          if (thread_idx >= remaining) {
              break;
          }
          int linear_idx = thread_idx + block_work_size * idx;
          store<${result_type}>(out[j], data[0], linear_idx);
          thread_idx += num_threads;
        }
      } else {
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
          out[j] = ${name}<${compute_type}>(${args});
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
)ESCAPE";

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
    int& major,
    int& minor,
    bool& compile_to_sass) {
  using CudaVersion = std::pair<int, int>;
  CudaVersion nvrtc_version;
  AT_CUDA_NVRTC_CHECK(
      nvrtc().nvrtcVersion(&nvrtc_version.first, &nvrtc_version.second));

  TORCH_CHECK(
      nvrtc_version.first >= 6,
      "NVRTC versions less than 6 are not supported. Is: ",
      nvrtc_version.first);

  // Version supported by device
  // Usually any lower version works too but is less efficient
  const CudaVersion dev_version = CudaVersion(prop->major, prop->minor);
  // Maximum version supported by the driver, cap dev_version to this
  CudaVersion max_dev_version;
  if (nvrtc_version.first <= 7) { // 7 supports 2-5.x
    max_dev_version = CudaVersion(5, 0);
  } else if (nvrtc_version.first <= 8) { // 8 supports 2-6.x
    max_dev_version = CudaVersion(6, 0);
  } else if (nvrtc_version.first <= 9) { // 9 supports 3-7.2
    max_dev_version = CudaVersion(7, 2);
  } else if (nvrtc_version.first <= 10) { // 10 supports 3-7.5
    max_dev_version = CudaVersion(7, 5);
  } else if (nvrtc_version == CudaVersion(11, 0)) { // 11.0 supports 3-8.0
    max_dev_version = CudaVersion(8, 0);
  } else {
    // If the driver version is unknown (i.e. newer than this code)
    // assume the driver supports this device
    max_dev_version = dev_version;
  }
  if (dev_version > max_dev_version) {
    major = max_dev_version.first;
    minor = max_dev_version.second;
    // if we are clamping major/minor, sass is not compatible
    compile_to_sass = false;
  } else {
    major = dev_version.first;
    minor = dev_version.second;
    compile_to_sass = true;
  }
}

//TODO another copy paste from jit, refactor so it's usable from both
void __inline__ initializeCudaContext() {
  // lazily construct context if non-existing yet;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  CUcontext pctx = nullptr;
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(nullptr);
  }
}

//FIXME - this are defined in Loops.cuh, but including Loops.cuh here would lead to circular includes Loops.cuh -> CUDALoops.cuh -> jit_utils.h -> Loops.cuh
#define THREAD_WORK_SIZE 4
constexpr int thread_work_size = THREAD_WORK_SIZE;

std::string generate_code(
    int nTensors,
    const std::string& func,
    const std::string& name,
    const std::string& f_inputs_type,
    const std::string& compute_type,
    const std::string& result_type,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    bool vectorized,
    int vec_size) {
  TemplateEnv env;
  env.s("index_type", "unsigned int");
  const int nInputs = nTensors - 1;
  env.s("nInputs", std::to_string(nInputs));
  env.s("scalar_type", f_inputs_type);
  env.s("compute_type", compute_type);
  env.s("functor", func);
  env.s("name", name);
  std::stringstream declare_load_arrays;
  for (int i = 0; i < nInputs; i++) {
    // TODO these arrays are potentially of the different types, use function
    // traits to determine the types
    declare_load_arrays << f_inputs_type << " arg" << std::to_string(i)
                        << "[" << std::to_string(thread_work_size) << "];\n";
  }
  env.s("declare_load_arrays", declare_load_arrays.str());
  std::stringstream declare_store_arrays;
  declare_store_arrays << result_type << " out"
                       << "[" << std::to_string(thread_work_size) << "];\n";
  env.s("declare_store_arrays", declare_store_arrays.str());
  const int nOutputs = 1; // FIXME
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
  if (f_inputs_type == "at::Half" || result_type == "at::Half" || dynamic_casting) {
    env.s("half_string", jiterator_half_support_literal);
  } else {
    env.s("half_string", "");
  }
  if (f_inputs_type == "at::BFloat16" || result_type == "at::BFloat16" || dynamic_casting) {
    env.s("bfloat16_string", jiterator_bfloat16_support_literal);
  } else {
    env.s("bfloat16_string", "");
  }
  if (dynamic_casting) {
    env.s("complex_string", jiterator_complex_support_literal);
  } else {
    env.s("complex_string", "");
  }

  if (!vectorized) {
    if (!dynamic_casting) {
      env.s("loader", "LoadWithoutCast");
      env.s("storer", "StoreWithoutCast");
    } else {
      env.s(
          "loader", std::string("LoadWithCast<" + std::to_string(nInputs) + ">"));
      env.s("storer", "StoreWithCast");
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
    store_outputs << "s.store<" << result_type
                  << ">(out[j], data[0], output_offsets[0]);\n";
    env.s("store_outputs", store_outputs.str());
    static auto cuda_template = CodeTemplate(jit_common_types + jit_code_template);
    return cuda_template.format(env);
  }

  // vectorized case
  env.s("vec_size", std::to_string(vec_size));
  env.s("result_type", result_type);
  std::stringstream vector_pointers;
  for (const auto i : c10::irange(nInputs)){
    auto i_string = std::to_string(i);
    vector_pointers << "vec_t_input * vec" << i_string <<
    " = reinterpret_cast<vec_t_input *>(data[" << i_string << "+1])" <<
    " + block_work_size / vec_size * idx;\n";
  }
  env.s("vector_pointers", vector_pointers.str());
  std::stringstream load_vectorized_inputs;
  for (const auto i : c10::irange(nInputs)) {
    auto i_string = std::to_string(i);
    load_vectorized_inputs << "v = vec" << i_string << "[thread_idx];\n";
    load_vectorized_inputs << "#pragma unroll\n";
    load_vectorized_inputs << "for (int j=0; j < vec_size; j++){\n";
    load_vectorized_inputs << "  arg" << i_string << "[vec_size * i + j] = v.val[j];\n";
    load_vectorized_inputs << "}\n";
  }
  env.s("load_vectorized_inputs", load_vectorized_inputs.str());
  std::stringstream load_unrolled_inputs;
  for (const auto i: c10::irange(nInputs)){
    auto i_string = std::to_string(i);
    load_unrolled_inputs << "arg" << i_string << "[j] = load<" << f_inputs_type
      << ">(data[" << std::to_string(i + nOutputs) << "], linear_idx);\n";
  }
  env.s("load_unrolled_inputs", load_unrolled_inputs.str());

  static auto cuda_template = CodeTemplate(jit_common_types + jit_vectorized_code_template);
  return cuda_template.format(env);
}

// Compiles the kernel
NvrtcFunction jit_pwise_function(
    const std::string& code,
    const std::string& kernel_name) {
  // Acquires device and NVRTC properties (for compile arch and occupancy calculations)
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int major = 0, minor = 0;
  bool compile_to_sass = false;
  codegenOutputQuery(prop, major, minor, compile_to_sass);
  // Creates the NVRTC program
  nvrtcProgram program;
  const auto& nvrtc = at::globalContext().getNVRTC();
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));
  // constructs nvrtc build arguments
#if defined(CUDA_VERSION) && CUDA_VERSION < 11010
  // compile to sass is not allowed prior to CUDA 11.1
  compile_to_sass = false;
#endif
  // CUDA 11.1 allows going directly to SASS (sm_) instead of PTX (compute_)
  // which gives better backwards compatibility to work on older driver,
  // (since older driver doesn't necessrily recognize PTX emitted by new
  // toolkit);
  // Meanwhile, for forward compatibility (future device with
  // `unsupported_arch==True`), since SASS are not necessarily compatible,
  // we fallback to PTX instead.
  const std::string compute = std::string("--gpu-architecture=") +
      (compile_to_sass ? "sm_" : "compute_") + std::to_string(major) +
      std::to_string(minor);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<const char*> args = {
      "--std=c++14", compute.c_str(), "-default-device"};

#ifndef NDEBUG
  // Add line info to generated kernels
  args.push_back("-lineinfo");
#else
  // Avoid excessive register usage from assertion
  args.push_back("-DNDEBUG");
#endif
  // compiles and validates result
  initializeCudaContext();
  const auto compilation_result =
      nvrtc.nvrtcCompileProgram(program, args.size(), args.data());
  if (compilation_result != NVRTC_SUCCESS) {
    size_t logsize;
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLogSize(program, &logsize));
    std::vector<char> log(logsize);
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLog(program, log.data()));
    std::stringstream cu;
    cu << log.data();
    throw std::runtime_error(cu.str() + code);
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

    NvrtcFunction compiled_kernel_;
    AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&(compiled_kernel_.module), ptx.data()));
    std::string name = kernel_name + "_kernel";
    AT_CUDA_DRIVER_CHECK(
        nvrtc.cuModuleGetFunction(&(compiled_kernel_.function), compiled_kernel_.module, name.c_str()));
    // TODO: use guards to avoid leaking
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcDestroyProgram(&program));

    return compiled_kernel_;
}

// TODO: may need/want to initialize CUDA context here (refactor into nvrtc call)
void launch_jitted_pwise_function(
    NvrtcFunction function,
    std::array<void*, 7>& args,
    const int nBlocks,
    const int kBlockSize) {
  initializeCudaContext();
  const auto& nvrtc = at::globalContext().getNVRTC();
  // Launches kernel on current stream
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_DRIVER_CHECK(nvrtc.cuLaunchKernel(
    function.function,
    nBlocks,
    1,
    1,
    kBlockSize,
    1,
    1,
    0,
    stream,
    args.data(),
    nullptr));
}

}}} // at::cuda::jit
