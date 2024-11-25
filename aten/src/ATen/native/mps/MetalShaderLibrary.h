#pragma once
#include <Metal/Metal.h>
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
  inline id<MTLComputePipelineState> getPipelineStateForFunc(
      const std::string& fname) {
    return getLibraryPipelineState(getLibrary(), fname).first;
  }
  id<MTLComputePipelineState> getPipelineStateForFunc(
      const std::string& fname,
      const std::initializer_list<std::string>& params) {
    return getLibraryPipelineState(getLibrary(params), fname).first;
  }
  inline id<MTLFunction> getMTLFunction(const std::string& fname) {
    return getLibraryPipelineState(getLibrary(), fname).second;
  }
  id<MTLFunction> getMTLFunction(
      const std::string& fname,
      const std::initializer_list<std::string>& params) {
    return getLibraryPipelineState(getLibrary(params), fname).second;
  }
  static MetalShaderLibrary& getBundledLibrary();

 protected:
  virtual id<MTLLibrary> getLibrary();
  virtual id<MTLLibrary> getLibrary(
      const std::initializer_list<std::string>& params);
  id<MTLLibrary> library = nil;

 private:
  std::pair<id<MTLComputePipelineState>, id<MTLFunction>>
  getLibraryPipelineState(id<MTLLibrary> lib, const std::string& fname);

  id<MTLLibrary> compileLibrary(const std::string& src);
  std::string shaderSource;
  unsigned nparams;
  MTLCompileOptions* compile_options;
  std::unordered_map<std::string, id<MTLLibrary>> libMap;
  std::unordered_map<
      std::string,
      std::pair<id<MTLComputePipelineState>, id<MTLFunction>>>
      cplMap;
};

} // namespace at::native::mps
