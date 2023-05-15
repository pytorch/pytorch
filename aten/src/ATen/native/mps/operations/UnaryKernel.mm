#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/UnaryConstants.h>
#include <torch/mps.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/erfinv_native.h>
#endif

#include <fmt/format.h>

namespace at::native {
const std::string& getMetalType(const c10::ScalarType& t) {
  // Mapping from c10::ScalarType to integral type that can be used for unary ops
  static std::unordered_map<c10::ScalarType, std::string> scalar_to_metal_type = {
      {c10::ScalarType::Half, "half"},
      {c10::ScalarType::Float, "float"},
      {c10::ScalarType::Long, "long"},
      {c10::ScalarType::Int, "int"},
      {c10::ScalarType::Short, "short"},
      {c10::ScalarType::Bool, "bool"},
      {c10::ScalarType::Char, "int8_t"},
  };

  auto it = scalar_to_metal_type.find(t);
  TORCH_CHECK(it != scalar_to_metal_type.end(), "Unsupported type ", t);
  return it->second;
}

const std::string& getMetalType(const Tensor& t) {
  return getMetalType(t.scalar_type());
}

const std::string& getMetalType(const c10::Scalar& s) {
  return getMetalType(s.type());
}
static inline id<MTLBuffer> getMTLBufferStorage(const Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static id<MTLLibrary> compileUnaryOpsLibrary(id<MTLDevice> device, const std::string& t1, const std::string& t2) {
  auto key = t1 + t2;
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  auto rc =
      [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(UNARY_KERNEL_TEMPLATE, t1, t2).c_str()]
                           options:options
                             error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
  libMap[key] = rc;
  return rc;
}

static id<MTLComputePipelineState> getCPLState(id<MTLDevice> device,
                                               const std::string& t1,
                                               const std::string& t2,
                                               const std::string& fname) {
  auto key = t1 + t2 + fname;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
  auto it = cplMap.find(key);
  if (it != cplMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  auto library = compileUnaryOpsLibrary(device, t1, t2);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func != nil, "Can't get function ", fname);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(
      rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key] = rc;
  return rc;
}

TORCH_IMPL_FUNC(erfinv_out_mps)(const Tensor& self, const Tensor& output_) {
  // handle erfinv ops using metal kernel
  // erfinv algorithm ported from aten/src/ATen/native/Math.h
  // https://github.com/pytorch/pytorch/blob/4154c8ea159fdaecc71ee9af820ac956193c875b/aten/src/ATen/native/Math.h#L152

  TORCH_CHECK(self.scalar_type() != ScalarType::Double, "MPS does not support erfinv op with scalar type: Double");

  Tensor outputTensor = output_;
  bool needs_output_copy = false;
  uint32_t length = output_.numel();
  if (length == 0) {
    return;
  }
  using namespace torch::mps;
  @autoreleasepool {
    Tensor inputTensor = self;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLComputePipelineState> cplState =
        getCPLState(device, getMetalType(outputTensor), getMetalType(self), "erfinv_mps_kernel");

    if (!self.is_contiguous()) {
      inputTensor = inputTensor.contiguous();
      outputTensor = outputTensor.contiguous();
      needs_output_copy = true;
      // calling continguous triggers a command encoder so need to commit it
      torch::mps::commit();
    }

    // Get a reference of the MPSStream MTLCommandBuffer.
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");
    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      id<MTLBuffer> outBuf = getMTLBufferStorage(outputTensor);
      id<MTLBuffer> inputBuf = getMTLBufferStorage(inputTensor);

      [computeEncoder setComputePipelineState:cplState];
      [computeEncoder setBuffer:outBuf offset:0 atIndex:0];
      [computeEncoder setBuffer:inputBuf offset:0 atIndex:1];
      // Set uniform buffer for the erfinv kernel polynomial constants
      struct ErfinvConstant uniforms;
      id<MTLBuffer> constantBuffer = [device newBufferWithLength:sizeof(ErfinvConstant)
                                                         options:MTLResourceStorageModePrivate];
      memcpy([constantBuffer contents], &uniforms, sizeof(ErfinvConstant));
      [computeEncoder setBuffer:constantBuffer offset:0 atIndex:2];

      MTLSize gridSize = MTLSizeMake(length, 1, 1);
      uint32_t maxThreadsPerGroup = [cplState maxTotalThreadsPerThreadgroup];

      NSUInteger threadsPerGroupSize = std::min(maxThreadsPerGroup, length);
      MTLSize threadGroupSize = MTLSizeMake(threadsPerGroupSize, 1, 1);
      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [computeEncoder endEncoding];
      torch::mps::commit();
    });
  }
  torch::mps::synchronize();
  if (needs_output_copy) {
    output_.copy_(outputTensor);
  }
}
} // namespace at::native