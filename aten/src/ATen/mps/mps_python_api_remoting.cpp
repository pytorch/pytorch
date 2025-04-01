#include <c10/macros/Export.h>

#include <ATen/native/mps/MetalShaderLibrary.h>

namespace at::native::mps {

  MetalShaderLibrary& MetalShaderLibrary::getBundledLibrary() {
    static auto l = MetalShaderLibrary("hell");
    return l;
  }

  std::vector<std::string> MetalShaderLibrary::getFunctionNames() {
    return std::vector<std::string>();
  }

  std::shared_ptr<MetalKernelFunction> MetalShaderLibrary::getKernelFunction(
    const std::string& name) {
    return nullptr;
  }

  MTLLibrary_t MetalShaderLibrary::compileLibrary(const std::string& src) {
    return nullptr;
  }

  std::pair<MTLComputePipelineState_t, MTLFunction_t> MetalShaderLibrary::getLibraryPipelineState(
      MTLLibrary_t lib,
      const std::string& fname) {
    MTLComputePipelineState_t a;
    MTLFunction_t b;

    return std::pair<MTLComputePipelineState_t, MTLFunction_t>(a, b);
  }

  MetalShaderLibrary::~MetalShaderLibrary() {}

  DynamicMetalShaderLibrary::~DynamicMetalShaderLibrary() {}

  MTLLibrary_t MetalShaderLibrary::getLibrary(const std::initializer_list<std::string>& params) {
    return nullptr;
  }

  MTLLibrary_t MetalShaderLibrary::getLibrary() {
    return nullptr;
  }

  // Constructor: Initializes the MetalKernelFunction with a compute pipeline state
  MetalKernelFunction::MetalKernelFunction(MTLComputePipelineState_t cps_) {

  }

  // Destructor: Cleans up any resources used by the MetalKernelFunction
  MetalKernelFunction::~MetalKernelFunction() {

  }

  // Shader properties
  uint64_t MetalKernelFunction::getMaxThreadsPerThreadgroup() const {
    return 0;
  }

  uint64_t MetalKernelFunction::getThreadExecutionWidth() const {
    return 0;
  }

  uint64_t MetalKernelFunction::getStaticThreadGroupMemoryLength() const {
    return 0;
  }

  // Executes a command block, typically on the Metal GPU
  void MetalKernelFunction::runCommandBlock(std::function<void(void)> f) {

  }

  // Start encoding a command, invoked from within runCommandBlock
  void MetalKernelFunction::startEncoding(void) {

  }

  // Set argument for the kernel function with the specified index and tensor
  void MetalKernelFunction::setArg(unsigned idx, const at::TensorBase& t) {

  }

  // Set argument for the kernel function with the specified index and raw pointer to data
  void MetalKernelFunction::setArg(unsigned idx, const void* ptr, uint64_t size) {

  }

  void MetalKernelFunction::dispatch(
    uint64_t length,
    std::optional<uint64_t> groupSize) {

  }

  void MetalKernelFunction::dispatch(
    c10::ArrayRef<uint64_t> length,
    c10::OptionalArrayRef<uint64_t> groupSize) {

  }

} // namespace at::native::mps
