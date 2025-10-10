#pragma once

#include <string>

#include <c10/util/irange.h>
#include <ATen/jit_macros.h>
#include <ATen/cuda/detail/LazyNVRTC.h>

namespace at::cuda::jit {

enum class BinaryFuncVariant {NoScalar, RhsScalar, LhsScalar};

struct NvrtcFunction {
  CUmodule module = CUmodule();
  CUfunction function = nullptr;
};

struct KernelDescriptor {
  std::string name;
  std::string f;
  c10::ScalarType f_inputs_type;
  c10::ScalarType result_type;
  c10::SmallVector<c10::ScalarType> extra_args_types;
  int nInputs, nOutputs;
};

// Helper function to return a vector<string>
// corresponding to the type of the arguments in parameter pack.
template <typename... Args>
c10::SmallVector<at::ScalarType> get_extra_args_types() {
  return {c10::CppTypeToScalarType<Args>::value ...};
}

template <
  typename result_type,
  typename f_inputs_type,
  typename... ExtraArgs>
KernelDescriptor make_kernel_descriptor(
    std::string name,
    std::string f,
    int nInputs,
    int nOutputs) {
  KernelDescriptor ret;
  ret.name = std::move(name);
  ret.f = std::move(f);
  ret.f_inputs_type = c10::CppTypeToScalarType<f_inputs_type>::value;
  ret.result_type = c10::CppTypeToScalarType<result_type>::value;
  ret.extra_args_types = get_extra_args_types<ExtraArgs...>();
  ret.nInputs = nInputs;
  ret.nOutputs = nOutputs;
  return ret;
}

inline int can_vectorize_up_to(size_t default_alignment, void *pointer) {
  auto ip = reinterpret_cast<uintptr_t>(pointer);
#ifdef USE_ROCM
  if ((default_alignment == 1) && (ip % (16 * default_alignment) == 0)) {
    return 16;
  }
  if ((default_alignment <= 2) && (ip % (8 * default_alignment) == 0)) {
    return 8;
  }
#else
  if (ip % (8 * default_alignment) == 0) {
    return 8;
  }
#endif
  if (ip % (4 * default_alignment) == 0) {
    return 4;
  }
  if (ip % (2 * default_alignment) == 0) {
    return 2;
  }
  return 1;
}

inline int can_vectorize_up_to(const KernelDescriptor &desc, c10::ArrayRef<char*> pointers) {
  TORCH_INTERNAL_ASSERT(desc.nOutputs == 1);
  TORCH_INTERNAL_ASSERT(static_cast<int64_t>(pointers.size()) == 1 + desc.nInputs);

  // Deals with output
  auto result_size = c10::scalarTypeToTypeMeta(desc.result_type).itemsize();
  auto result = can_vectorize_up_to(result_size, pointers[0]);

  // Incorporates input(s)
  auto input_size = c10::scalarTypeToTypeMeta(desc.f_inputs_type).itemsize();
  for (auto i : c10::irange(1, pointers.size())) {
    result = std::min(result, can_vectorize_up_to(input_size, pointers[i]));
  }

  return result;
}

//FIXME - this are defined in Loops.cuh, but including Loops.cuh here would lead to circular includes Loops.cuh -> CUDALoops.cuh -> jit_utils.h -> Loops.cuh
#ifdef USE_ROCM
#define JIT_THREAD_WORK_SIZE 4
#else
#define JIT_THREAD_WORK_SIZE 8
#endif

int calc_io_size(
    const int nInputs,
    const int nOutputs,
    const c10::ScalarType& inputs_type,
    const c10::ScalarType& result_type);

int calc_thread_work_size(
    const int nInputs,
    const int nOutputs,
    const c10::ScalarType& inputs_type,
    const c10::ScalarType& result_type);

std::string generate_code(
    int nInputs,
    int nOutputs,
    const std::string& func,
    const std::string& name,
    const std::string& f_inputs_type,
    const std::string& compute_type,
    const std::string& result_type,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    c10::SmallVector<std::string>& extra_args_typenames,
    int thread_work_size=JIT_THREAD_WORK_SIZE,
    bool vectorized=false,
    int vec_size=0,
    bool return_by_ref=false);

std::string generate_code(
    const KernelDescriptor &desc,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    int thread_work_size=JIT_THREAD_WORK_SIZE,
    bool vectorized=false,
    int vec_size=0,
    bool return_by_ref=false);

std::string generate_reduction_code(
    int nOutputs,
    const std::string& func,
    const std::string& name,
    const int vt0,
    const std::string& f_inputs_type,
    const std::string& reduction_accum_type,
    const std::string& result_type,
    bool contiguous,
    bool vectorized,
    int vec_size,
    int max_threads_codegen);

std::string generate_reduction_code(
    const KernelDescriptor &desc,
    const int vt0,
    bool contiguous,
    bool vectorized,
    int vec_size,
    int max_threads_codegen);

NvrtcFunction jit_pwise_function(
    const std::string& code,
    const std::string& kernel_name);

void launch_jitted_pwise_function(
    NvrtcFunction function,
    const void* args[],
    const dim3 nBlocks,
    const dim3 kBlockSize,
    const int smem=0);

template <typename T>
struct delayed_false : std::false_type {
};

// Defines type names
// NOTE: General case is instantiated only for invalid types.
// All the valid types have specialization using the TYPE_NAME_FN
// macro below.
template <typename T>
inline std::string typeName() {
  // we can't use static_assert(false) directly as the
  // program will be not compiled even if the template is not
  // instantiated, so we use `delayed_false`
  // to make sure compiler doesn't eagerly raise
  // fail this assertion.
  static_assert(delayed_false<T>::value, "invalid type for jiterator");
  return "void";
}

#define TYPE_NAME_FN(ctype, name) \
template <> inline std::string typeName<ctype>(){ \
    return std::string(#ctype);    \
}

AT_FORALL_SCALAR_TYPES(TYPE_NAME_FN)
#undef TYPE_NAME_FN
// JIT uses std::complex directly, because nvRTC compile programs
// with -default-device, so there is no such issue like:
//   "std::sin(complex) is __host__ only"
template <> inline std::string typeName<bool>(){
    return "bool";
}
template <> inline std::string typeName<c10::complex<at::Half>>(){
    return "std::complex<at::Half>";
}
template <> inline std::string typeName<c10::complex<float>>(){
    return "std::complex<float>";
}
template <> inline std::string typeName<c10::complex<double>>(){
    return "std::complex<double>";
}
template <> inline std::string typeName<at::Half>(){
    return "at::Half";
}
template <> inline std::string typeName<at::BFloat16>(){
    return "at::BFloat16";
}
template <> inline std::string typeName<at::Float8_e5m2>(){
    return "at::Float8_e5m2";
}
template <> inline std::string typeName<at::Float8_e4m3fn>(){
    return "at::Float8_e4m3fn";
}
template <> inline std::string typeName<at::Float8_e5m2fnuz>() {
    return "at::Float8_e5m2fnuz";
}
template <> inline std::string typeName<at::Float8_e4m3fnuz>() {
    return "at::Float8_e4m3fnuz";
}
template <> inline std::string typeName<at::Float8_e8m0fnu>() {
    // TODO(#146647): Can the code here be made generic for any scalartype?
    return "at::Float8_e8m0fnu";
}

#define TYPE_NAME_CASE(ctype, scalartype)                    \
  case ScalarType::scalartype:  return typeName<ctype>();
inline std::string typeName(ScalarType t) {
    switch (t) {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(TYPE_NAME_CASE)
      default:
          TORCH_CHECK(false, "invalid type for jiterator");
    }
}
#undef TYPE_NAME_CASE

TORCH_CUDA_CPP_API void initializeCudaContext();

} // namespace at::cuda::jit
