#pragma once
#include <c10/macros/Macros.h>
#ifdef __OBJC__
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
#include <Metal/Metal.h>
C10_DIAGNOSTIC_POP()
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLFunction> MTLFunction_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
typedef id<MTLBuffer> MTLBuffer_t;
#else
typedef void MTLCompileOptions;
typedef void* MTLLibrary_t;
typedef void* MTLFunction_t;
typedef void* MTLComputePipelineState_t;
typedef void* MTLComputeCommandEncoder_t;
typedef void* MTLBuffer_t;
#endif

#include <c10/core/Scalar.h>
#include <c10/util/OptionalArrayRef.h>
#include <functional>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Forward declaration of TensorBase and TensorIteratorBase
namespace at {
class TensorBase;
struct TensorIteratorBase;
} // namespace at

namespace at::native::mps {

namespace detail {
template <typename T>
class has_size_type {
  template <typename U>
  static constexpr std::true_type check(typename U::size_type*);
  template <typename>
  static constexpr std::false_type check(...);

 public:
  static constexpr bool value = decltype(check<T>(nullptr))::value;
};

template <typename T>
constexpr bool has_size_type_v = has_size_type<T>::value;

} // namespace detail

// Returns `gpuAddress` of respective `id<MTLBuffer>` plus storage offset
void* get_tensor_gpu_address(const at::TensorBase&);

class MetalKernelFunction {
 public:
  MetalKernelFunction(MTLComputePipelineState_t cps_, MTLFunction_t f_);
  ~MetalKernelFunction();
  MetalKernelFunction(MetalKernelFunction&) = delete;
  // Shader properties
  uint64_t getMaxThreadsPerThreadgroup() const;
  uint64_t getThreadExecutionWidth() const;
  uint64_t getStaticThreadGroupMemoryLength() const;
  void runCommandBlock(std::function<void(void)> f);
  // Methods below should be called from runCommandBlock function
  void startEncoding();
  void setArg(unsigned idx, const at::TensorBase& t);
  void setArg(unsigned idx, const void* ptr, uint64_t size);
  void setErrorBufferIndex(unsigned idx);
  template <
      typename T,
      typename = std::enable_if_t<
          std::is_integral_v<T> || std::is_same_v<T, float> ||
          (std::is_class_v<T> && std::is_trivially_copyable_v<T> &&
           !detail::has_size_type_v<T>)>>
  inline void setArg(unsigned idx, const T val) {
    setArg(idx, &val, sizeof(T));
  }

  template <
      typename Container,
      typename = std::enable_if_t<detail::has_size_type_v<Container>>>
  inline void setArg(unsigned idx, const Container& values) {
    setArg(
        idx,
        values.data(),
        values.size() * sizeof(typename Container::value_type));
  }
  void dispatch(
      uint64_t length,
      std::optional<uint64_t> groupSize = std::nullopt);
  void dispatch(
      c10::ArrayRef<uint64_t> length,
      c10::OptionalArrayRef<uint64_t> groupSize = std::nullopt);

 private:
  MTLComputePipelineState_t cps;
  MTLFunction_t func;
  MTLComputeCommandEncoder_t encoder = nullptr;
};

class MetalShaderLibrary {
 public:
  MetalShaderLibrary(std::string src)
      : shaderSource(std::move(src)), nparams(0), compile_options(nullptr) {}
  MetalShaderLibrary(std::string src, unsigned nparams_)
      : shaderSource(std::move(src)),
        nparams(nparams_),
        compile_options(nullptr) {}
  MetalShaderLibrary(
      std::string src,
      unsigned nparams_,
      MTLCompileOptions* compile_options_)
      : shaderSource(std::move(src)),
        nparams(nparams_),
        compile_options(compile_options_) {}
  MetalShaderLibrary(const MetalShaderLibrary&) = delete;
  virtual ~MetalShaderLibrary();
  std::vector<std::string> getFunctionNames();
  std::shared_ptr<MetalKernelFunction> getKernelFunction(
      const std::string& name);
  // Returns a raw pointer to the kernel function for use in C APIs
  MetalKernelFunction* getCachedKernelFunctionPtr(const std::string& name);
  inline MTLComputePipelineState_t getPipelineStateForFunc(
      const std::string& fname) {
    return getLibraryPipelineState(getLibrary(), fname).first;
  }
  MTLComputePipelineState_t getPipelineStateForFunc(
      const std::string& fname,
      const std::initializer_list<std::string>& params) {
    return getLibraryPipelineState(getLibrary(params), fname).first;
  }
  inline MTLFunction_t getMTLFunction(const std::string& fname) {
    return getLibraryPipelineState(getLibrary(), fname).second;
  }
  MTLFunction_t getMTLFunction(
      const std::string& fname,
      const std::initializer_list<std::string>& params) {
    return getLibraryPipelineState(getLibrary(params), fname).second;
  }
  static MetalShaderLibrary& getBundledLibrary();
  // Returns whether the library exposes a kernel under the given host_name.
  bool hasFunction(const std::string& fname);
  // Dispatch is probe-then-fallback for non-alpha calls: first try the direct
  // per-(out,in) kernel; if it isn't registered, fall back to the
  // `_dense_castout_<in>` / `_strided_castout_<in>` variant registered for the
  // input dtype by REGISTER_UNARY_OP. The castout kernel computes the functor
  // in the input dtype and casts the result to the user-supplied output dtype
  // on store, matching CPU semantics. Alpha kernels skip the fallback (no
  // alpha castout variants are registered).
  void exec_unary_kernel(
      TensorIteratorBase& iter,
      const std::string& name,
      const std::optional<c10::Scalar> alpha = std::nullopt,
      const std::optional<c10::ScalarType> scalar_arg_type = std::nullopt,
      const std::optional<uint32_t> ilp_threshold = std::nullopt);
  // Raw cross-dtype copy variant for call sites that don't have a
  // TensorIterator -- e.g. when the destination is a Metal-wrapped CPU buffer
  // (newBufferWithBytesNoCopy) from a copy_from_mps_ path. Always takes the
  // castout fallback (cross-dtype is the use case); contiguity is implied. The
  // kernel name must be one of the registered cast-capable unary ops
  // (copy_identity, copy_conj, copy_neg, copy_conj_neg). offsets are in bytes;
  // numel is the element count to process.
  void exec_unary_kernel_raw(
      const std::string& name,
      MTLBuffer_t src_buf,
      uint32_t src_offs_bytes,
      c10::ScalarType src_dtype,
      MTLBuffer_t dst_buf,
      uint32_t dst_offs_bytes,
      c10::ScalarType dst_dtype,
      uint32_t numel,
      const std::optional<uint32_t> ilp_threshold = std::nullopt);
  // `ilp_threshold` lets callers tune when the dense ILP variant kicks in
  // (numel >= threshold). When unspecified, the default is the same 256K
  // crossover used by the unary path, but only for floating-point output;
  // non-float outputs get UINT32_MAX (i.e. ILP off by default). Comparison
  // and other ops with different memory-bandwidth profiles can override.
  // `natural_output_dtype` is the dtype the kernel naturally produces (its
  // registered DTYPEO). Defaults to `iter.common_dtype()`, which is right for
  // arithmetic kernels where DTYPEO==compute precision. Comparison kernels
  // produce bool and must pass `kBool` so the output-cast fallback allocates
  // the right temp.
  void exec_binary_kernel(
      TensorIteratorBase& iter,
      const std::string& name,
      const std::optional<c10::Scalar> alpha = std::nullopt,
      const std::optional<c10::ScalarType> scalar_arg_type = std::nullopt,
      const std::optional<c10::ScalarType> natural_output_dtype = std::nullopt,
      const std::optional<uint32_t> ilp_threshold = std::nullopt);
  void exec_ternary_kernel(TensorIteratorBase& iter, const std::string& name);

  template <typename T>
  void exec_unary_kernel_with_params(
      TensorIteratorBase& iter,
      const std::string& name,
      T params,
      const std::string& params_type_name);
  template <typename T>
  void exec_binary_kernel_with_params(
      TensorIteratorBase& iter,
      const std::string& name,
      T params,
      const std::string& params_type_name);

 protected:
  virtual MTLLibrary_t getLibrary();
  virtual MTLLibrary_t getLibrary(
      const std::initializer_list<std::string>& params);
  MTLLibrary_t library = nullptr;

 private:
  std::pair<MTLComputePipelineState_t, MTLFunction_t> getLibraryPipelineState(
      MTLLibrary_t lib,
      const std::string& fname);
  MTLLibrary_t compileLibrary(const std::string& src);
  std::string shaderSource;
  unsigned nparams;
  MTLCompileOptions* compile_options;
  std::unordered_map<std::string, MTLLibrary_t> libMap;
  std::unordered_map<
      std::string,
      std::pair<MTLComputePipelineState_t, MTLFunction_t>>
      cplMap;
  // Cache for kernel functions returned by getCachedKernelFunctionPtr
  std::unordered_map<std::string, std::unique_ptr<MetalKernelFunction>>
      kernelCache;
  // Lazily populated set of all kernel host_names in the library; used by
  // hasFunction() to answer probes from exec_unary_kernel.
  std::unordered_set<std::string> functionNames;
  bool functionNamesPopulated = false;
};

class DynamicMetalShaderLibrary : public MetalShaderLibrary {
 public:
  DynamicMetalShaderLibrary(const std::string& src) : MetalShaderLibrary(src) {
    // Compile right away
    getLibrary();
  }
  ~DynamicMetalShaderLibrary() override;
};

class PrecompiledMetalShaderLibrary : public MetalShaderLibrary {
 public:
  explicit PrecompiledMetalShaderLibrary(std::vector<uint8_t> data);
  explicit PrecompiledMetalShaderLibrary(const std::string& path);
  ~PrecompiledMetalShaderLibrary() override;
};

} // namespace at::native::mps
