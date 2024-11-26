#pragma once
#ifdef __OBJC__
#include <Metal/Metal.h>
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLFunction> MTLFunction_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
#else
typedef void MTLCompileOptions;
typedef void* MTLLibrary_t;
typedef void* MTLFunction_t;
typedef void* MTLComputePipelineState_t;
typedef void* MTLComputeCommandEncoder_t;
#endif

#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace at {
class TensorBase;
}

namespace at::native::mps {

class MetalKernelFunction {
 public:
  MetalKernelFunction(MTLComputePipelineState_t cps_) : cps(cps_) {}
  MetalKernelFunction(MetalKernelFunction&) = delete;
  ~MetalKernelFunction();
  // Shader properties
  uint64_t getMaxThreadsPerThreadgroup() const;
  uint64_t getThreadExecutionWidth() const;
  uint64_t getStaticThreadGroupMemoryLength() const;
  void runCommandBlock(std::function<void(void)> f);
  // Methods below should be called from runCommandBlock functionT
  void startEncoding();
  void setArg(unsigned idx, const at::TensorBase& t);
  void setArg(unsigned idx, const void* ptr, uint64_t size);
  void dispatch(
      uint64_t length,
      std::optional<uint64_t> groupSize = std::nullopt);
  void dispatch(
      std::array<uint64_t, 2> length,
      std::optional<std::array<uint64_t, 2>> groupSize = std::nullopt);

 private:
  MTLComputePipelineState_t cps;
  MTLComputeCommandEncoder_t encoder = nullptr;
};

class MetalShaderLibrary {
 public:
  MetalShaderLibrary(const std::string& src)
      : shaderSource(src), nparams(0), compile_options(nullptr) {}
  MetalShaderLibrary(const std::string& src, unsigned nparams_)
      : shaderSource(src), nparams(nparams_), compile_options(nullptr) {}
  MetalShaderLibrary(
      const std::string& src,
      unsigned nparams_,
      MTLCompileOptions* compile_options_)
      : shaderSource(src),
        nparams(nparams_),
        compile_options(compile_options_) {}
  MetalShaderLibrary(const MetalShaderLibrary&) = delete;
  virtual ~MetalShaderLibrary() = default;
  std::vector<std::string> getFunctionNames();
  std::shared_ptr<MetalKernelFunction> getKernelFunction(
      const std::string& name);
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
};

class DynamicMetalShaderLibrary : public MetalShaderLibrary {
 public:
  DynamicMetalShaderLibrary(const std::string& src) : MetalShaderLibrary(src) {
    // Compile right away
    getLibrary();
  }
};

} // namespace at::native::mps
