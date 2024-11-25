#pragma once
#ifdef __OBJC__
#include <Metal/Metal.h>
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLFunction> MTLFunction_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
#else
typedef void MTLCompileOptions;
typedef void* MTLLibrary_t;
typedef void* MTLFunction_t;
typedef void* MTLComputePipelineState_t;
#endif

#include <unordered_map>
#include <vector>

namespace at::native::mps {
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
