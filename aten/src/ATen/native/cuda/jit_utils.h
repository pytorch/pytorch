#pragma once

#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <c10/util/irange.h>
#include <ATen/jit_macros.h>
#include <ATen/cuda/detail/LazyNVRTC.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

namespace at { namespace cuda { namespace jit {

enum class BinaryFuncVariant {NoScalar, RhsScalar, LhsScalar};

struct NvrtcFunction {
  CUmodule module = CUmodule();
  CUfunction function = nullptr;
};

std::string generate_code(
    int nInputs,
    int nOutputs,
    const std::string& func,
    const std::string& name,
    const std::string& f_input_type,
    const std::string& compute_type,
    const std::string& result_type,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    c10::SmallVector<std::string>& extra_args_typenames,
    bool vectorized=false,
    int vec_size=0);

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

NvrtcFunction jit_pwise_function(
    const std::string& code,
    const std::string& kernel_name);

void launch_jitted_pwise_function(
    NvrtcFunction function,
    void* args[],
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

#define TYPE_NAME_CASE(ctype, scalartype)                    \
  case ScalarType::scalartype:  return std::string(#ctype);
inline std::string typeName(ScalarType t) {
    switch (t) {
        AT_FORALL_SCALAR_TYPES(TYPE_NAME_CASE)
        case ScalarType::Bool : return "bool";
        case ScalarType::Half : return "at::Half";
        case ScalarType::BFloat16 : return "at::BFloat16";
        case ScalarType::ComplexFloat : return "std::complex<float>";
        case ScalarType::ComplexDouble : return "std::complex<double>";
        default:
            TORCH_CHECK(false, "invalid type for jiterator");
    }
}
#undef TYPE_NAME_CASE

}}}  // namespace at::cuda::jit
