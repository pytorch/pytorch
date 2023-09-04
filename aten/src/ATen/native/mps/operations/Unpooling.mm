#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/max_unpool2d_native.h>
#endif

namespace at::native {
namespace mps {
static const char* UNPOOL_OPS_TEMPLATE = R"METAL(

kernel void max_unpooling2d_forward(constant int64_t& numInputElements [[buffer(0)]],
                                    device {0} *input [[buffer(1)]],
                                    device int64_t* indices [[buffer(2)]],
                                    constant int64_t& numChannels [[buffer(3)]],
                                    constant int64_t& inputHeight [[buffer(4)]],
                                    constant int64_t& inputWidth [[buffer(5)]],
                                    constant int64_t& outputHeight [[buffer(6)]],
                                    constant int64_t& outputWidth [[buffer(7)]],
                                    device {1} *output [[buffer(8)]],
                                    uint id [[thread_position_in_grid]]) {{
  int64_t outputImageSize = outputHeight * outputWidth;
  if (id < numInputElements) {{
    int c = (id / inputWidth / inputHeight) % numChannels;
    int n = id / inputWidth / inputHeight / numChannels;
    output += (n * numChannels + c) * outputHeight * outputWidth;
    int maxind = indices[id];
    if (maxind >= 0 && maxind < outputImageSize) {{
      output[maxind] = input[id];
    }}
  }}
}}

)METAL";

static id<MTLLibrary> compileUnpoolOpsLibrary(id<MTLDevice> device, const std::string& t1, const std::string& t2) {
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
      [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(UNPOOL_OPS_TEMPLATE, t1, t2).c_str()]
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
  auto library = compileUnpoolOpsLibrary(device, t1, t2);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func != nil, "Can't get function ", fname);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(
      rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key] = rc;
  return rc;
}

static void dispatch1DJob(id<MTLComputeCommandEncoder> commandEncoder,
                          id<MTLComputePipelineState> cplState,
                          uint32_t length) {
  uint32_t maxThreadsPerGroup = [cplState maxTotalThreadsPerThreadgroup];
  auto size = MTLSizeMake(length, 1, 1);
  auto threadGroupSize = MTLSizeMake(std::min(maxThreadsPerGroup, length), 1, 1);
  [commandEncoder dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
}

} // namespace mps

Tensor& max_unpooling2d_forward_out_mps(const Tensor& self_,
                                        const Tensor& indices_,
                                        IntArrayRef output_size,
                                        Tensor& output) {
  at::globalContext().alertNotDeterministic("max_unpooling2d_forward_out");

  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(indices_.scalar_type() == at::ScalarType::Long,
              "elements in indices should be type int64 but got: ",
              indices_.scalar_type());

  for (int64_t i = 1; i < self_.ndimension(); ++i) {
    TORCH_CHECK(self_.size(i) > 0,
                "max_unpooling2d_forward_out_cuda(): ",
                "Expected input to have non-zero size for non-batch dimensions, but got ",
                self_.sizes(),
                " with dimension ",
                i,
                " being empty.");
  }

  TORCH_CHECK((self_.ndimension() == 3 || self_.ndimension() == 4),
              "Input to max_unpooling2d should be a 3d or 4d Tensor, but got tensor with dimension: ",
              self_.ndimension());
  TORCH_CHECK(self_.sizes() == indices_.sizes(),
              "Expected shape of indices to be: ",
              self_.sizes(),
              " but got: ",
              indices_.sizes());
  TORCH_CHECK(output_size.size() == 2,
              "There should be exactly two elements (height, width) in output_size, but got ",
              output_size.size(),
              " elements.");

  auto outputHeight = output_size[0];
  auto outputWidth = output_size[1];

  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t numBatch = 1;

  auto self = self_.contiguous();
  auto indices = indices_.contiguous();

  if (self.ndimension() == 4) {
    numBatch = self.size(0);
    dimw++;
    dimh++;
  }

  int64_t numChannels = self.size(dimh - 1);
  int64_t inputHeight = self.size(dimh);
  int64_t inputWidth = self.size(dimw);
  int64_t numInputElements = self.numel();

  output.resize_({numBatch, numChannels, outputHeight, outputWidth});
  output.zero_();

  if (numInputElements != 0) {
    using namespace at::mps;

    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputePipelineState> cplState = mps::getCPLState(MPSDevice::getInstance()->device(),
                                                            mps::scalarToMetalTypeString(self.scalar_type()),
                                                            mps::scalarToMetalTypeString(output.scalar_type()),
                                                            "max_unpooling2d_forward");
    dispatch_sync(stream->queue(), ^() {
      getMPSProfiler().beginProfileKernel(cplState, "max_unpooling2d_forward", {self});

      id<MTLComputeCommandEncoder> commandEncoder = stream->commandEncoder();

      id<MTLBuffer> outBuf = __builtin_bit_cast(id<MTLBuffer>, output.storage().data());
      id<MTLBuffer> selfBuf = __builtin_bit_cast(id<MTLBuffer>, self.storage().data());
      id<MTLBuffer> indicesBuf = __builtin_bit_cast(id<MTLBuffer>, indices.storage().data());

      [commandEncoder pushDebugGroup:@"Dispatch max_unpooling2d_forward kernel"];
      [commandEncoder setComputePipelineState:cplState];

      [commandEncoder setBytes:&numInputElements length:sizeof(numInputElements) atIndex:0];
      [commandEncoder setBuffer:selfBuf offset:self.storage_offset() * self.itemsize() atIndex:1];
      [commandEncoder setBuffer:indicesBuf offset:indices.storage_offset() * indices.itemsize() atIndex:2];
      [commandEncoder setBytes:&numChannels length:sizeof(numChannels) atIndex:3];
      [commandEncoder setBytes:&inputHeight length:sizeof(inputHeight) atIndex:4];
      [commandEncoder setBytes:&inputWidth length:sizeof(inputWidth) atIndex:5];
      [commandEncoder setBytes:&outputHeight length:sizeof(outputHeight) atIndex:6];
      [commandEncoder setBytes:&outputWidth length:sizeof(outputWidth) atIndex:7];
      [commandEncoder setBuffer:outBuf offset:output.storage_offset() * output.itemsize() atIndex:8];

      mps::dispatch1DJob(commandEncoder, cplState, static_cast<uint32_t>(numInputElements));

      getMPSProfiler().endProfileKernel(cplState);
    });
  }
  if (self.ndimension() == 3) {
    output.resize_({numChannels, outputHeight, outputWidth});
  }
  return output;
}

Tensor max_unpooling2d_forward_mps(const Tensor& self, const Tensor& indices, IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  max_unpooling2d_forward_out_mps(self, indices, output_size, output);
  return output;
}

} // namespace at::native
