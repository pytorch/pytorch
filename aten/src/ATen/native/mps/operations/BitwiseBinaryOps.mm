#include <ATen/mps/MPSStream.h>
#include <ATen/native/Resize.h>
#include <fmt/format.h>
#include <torch/library.h>

namespace {
static const char* BITWISE_OPS_TEMPLATE = R"METAL(

kernel void bitwise_and_tensor(constant uint& length [[buffer(0)]],
                         device {0}  *out [[buffer(1)]],
                         device {1}  *a [[buffer(2)]],
                         device {2}  *b [[buffer(3)]],
                         uint offset [[thread_position_in_grid]]) {{
  if (offset >= length) {{
    return;
  }}
  out[offset] = a[offset] & b [offset];
}}

kernel void bitwise_and_scalar(constant uint& length [[buffer(0)]],
                         device {0}  *out [[buffer(1)]],
                         device {1}  *a [[buffer(2)]],
                         constant {2}  &b [[buffer(3)]],
                         uint offset [[thread_position_in_grid]]) {{
  if (offset >= length) {{
    return;
  }}
  out[offset] = a[offset] & b;
}}


kernel void bitwise_or_tensor(constant uint& length [[buffer(0)]],
                         device {0}  *out [[buffer(1)]],
                         device {1}  *a [[buffer(2)]],
                         device {2}  *b [[buffer(3)]],
                         uint offset [[thread_position_in_grid]]) {{
  if (offset >= length) {{
    return;
  }}
  out[offset] = a[offset] | b [offset];
}}

kernel void bitwise_or_scalar(constant uint& length [[buffer(0)]],
                         device {0}  *out [[buffer(1)]],
                         device {1}  *a [[buffer(2)]],
                         constant {2}  &b [[buffer(3)]],
                         uint offset [[thread_position_in_grid]]) {{
  if (offset >= length) {{
    return;
  }}
  out[offset] = a[offset] | b;
}}

kernel void bitwise_xor_tensor(constant uint& length [[buffer(0)]],
                         device {0}  *out [[buffer(1)]],
                         device {1}  *a [[buffer(2)]],
                         device {2}  *b [[buffer(3)]],
                         uint offset [[thread_position_in_grid]]) {{
  if (offset >= length) {{
    return;
  }}
  out[offset] = a[offset] ^ b [offset];
}}

kernel void bitwise_xor_scalar(constant uint& length [[buffer(0)]],
                         device {0}  *out [[buffer(1)]],
                         device {1}  *a [[buffer(2)]],
                         constant {2}  &b [[buffer(3)]],
                         uint offset [[thread_position_in_grid]]) {{
  if (offset >= length) {{
    return;
  }}
  out[offset] = a[offset] ^ b;
}}

)METAL";


const std::string& getMetalType(const c10::ScalarType& t) {
  // Mapping from c10::ScalarType to integral type that can be used for bitwise ops
  // As bitwise ops sign-agnostic map signed/unsigned char and boolean to the same type
  static std::unordered_map<c10::ScalarType, std::string> scalar_to_metal_type = {
    {c10::ScalarType::Long, "long"},
    {c10::ScalarType::Int, "int"},
    {c10::ScalarType::Short, "short"},
    {c10::ScalarType::Byte, "char"},
    {c10::ScalarType::Char, "char"},
    {c10::ScalarType::Bool, "char"},
  };

  auto it = scalar_to_metal_type.find(t);
  TORCH_CHECK(it != scalar_to_metal_type.end(), "Unsupported type ", t);
  return it->second;
}

const std::string& getMetalType(const at::Tensor& t) {
  return getMetalType(t.scalar_type());
}

const std::string& getMetalType(const c10::Scalar& s) {
  return getMetalType(s.type());
}


static id<MTLLibrary> compileBitwiseOpsLibrary(id<MTLDevice> device,
                                               const std::string& t1,
                                               const std::string& t2,
                                               const std::string& t3) {
  auto key = t1 + t2 + t3;
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError *error = nil;
  auto rc  = [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(BITWISE_OPS_TEMPLATE, t1, t2, t3).c_str()]
                                  options:nil
                                    error:&error];
 TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
 libMap[key] = rc;
 return rc;
}


static id<MTLComputePipelineState> getCPLState(id<MTLDevice> device,
                                                const std::string& t1,
                                                const std::string& t2,
                                                const std::string& t3,
                                                const std::string& fname) {
  auto key = t1 + t2 + t3 + fname;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
  auto it = cplMap.find(key);
  if (it != cplMap.end()) {
     return it->second;
  }
  NSError *error = nil;
  auto library = compileBitwiseOpsLibrary(device, t1, t2, t3);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func != nil, "Can't get function ", fname);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key]  = rc;
  return rc;
}

void handle_tensor_tensor_binary_op(const at::Tensor& self, const at::Tensor& other, at::Tensor& output, const std::string& kernel_name) {
  using namespace at::mps;
  MPSStream* stream = getCurrentMPSStream();
  id<MTLComputePipelineState> cplState = getCPLState(MPSDevice::getInstance()->device(),
                                                     getMetalType(output),
                                                     getMetalType(self),
                                                     getMetalType(other),
                                                     kernel_name);
  uint32_t length = output.numel();
  dispatch_sync(stream->queue(), ^(){
    id<MTLCommandBuffer> buffer = stream->commandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = [buffer computeCommandEncoder];

    id<MTLBuffer> outBuf = __builtin_bit_cast(id<MTLBuffer>, output.storage().data());
    id<MTLBuffer> selfBuf = __builtin_bit_cast(id<MTLBuffer>, self.storage().data());
    id<MTLBuffer> otherBuf = __builtin_bit_cast(id<MTLBuffer>, other.storage().data());

    [commandEncoder pushDebugGroup:[NSString stringWithFormat:@"Dispatch %s kernel", kernel_name.c_str()]];
    [commandEncoder setComputePipelineState:cplState];
    [commandEncoder setBytes:&length length:sizeof(length) atIndex:0];
    [commandEncoder setBuffer:outBuf offset:output.storage_offset()*output.itemsize() atIndex:1];
    [commandEncoder setBuffer:selfBuf offset:self.storage_offset()*self.itemsize()  atIndex:2];
    [commandEncoder setBuffer:otherBuf offset:other.storage_offset()*other.itemsize() atIndex:3];
    [commandEncoder dispatchThreadgroups:MTLSizeMake((length + 511) / 512, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    [commandEncoder endEncoding];
  });
}

void handle_tensor_scalar_binary_op(const at::Tensor& self, const at::Scalar& other, at::Tensor& output, const std::string& kernel_name) {
  using namespace at::mps;
  MPSStream* stream = getCurrentMPSStream();
  id<MTLComputePipelineState> cplState = getCPLState(MPSDevice::getInstance()->device(),
                                                     getMetalType(output),
                                                     getMetalType(self),
                                                     getMetalType(other),
                                                     kernel_name);
  uint64_t sval = other.to<int64_t>();
  uint32_t length = output.numel();
  dispatch_sync(stream->queue(), ^(){
    id<MTLCommandBuffer> buffer = stream->commandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = [buffer computeCommandEncoder];

    id<MTLBuffer> outBuf = __builtin_bit_cast(id<MTLBuffer>, output.storage().data());
    id<MTLBuffer> selfBuf = __builtin_bit_cast(id<MTLBuffer>, self.storage().data());

    [commandEncoder pushDebugGroup:[NSString stringWithFormat:@"Dispatch %s kernel", kernel_name.c_str()]];
    [commandEncoder setComputePipelineState:cplState];
    [commandEncoder setBytes:&length length:sizeof(length) atIndex:0];
    [commandEncoder setBuffer:outBuf offset:output.storage_offset()*output.itemsize() atIndex:1];
    [commandEncoder setBuffer:selfBuf offset:self.storage_offset()*self.itemsize()  atIndex:2];
    [commandEncoder setBytes:&sval length:sizeof(sval) atIndex:3];
    [commandEncoder dispatchThreadgroups:MTLSizeMake((length + 511) / 512, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    [commandEncoder endEncoding];
  });
}

at::Tensor& _bitwise_op_out_mps (const at::Tensor& self, const at::Tensor& other, at::Tensor& output_, const std::string& op_name) {
  using namespace at::mps;
  const bool is_self_scalar = self.dim() == 0;
  const bool is_other_scalar = other.dim() == 0;

  at::Tensor output = output_;
  bool needs_output_copy = false;

  if (!output_.is_contiguous()) {
    output = output_.contiguous();
    needs_output_copy = true;
  }
  auto output_size = at::infer_size_dimvector(self.sizes(), other.sizes());
  at::native::resize_output(output, output_size);
  if (is_other_scalar && is_self_scalar) {
    if (op_name == "and") {
      output.fill_(c10::Scalar(self.item<int64_t>() & other.item<int64_t>()));
    } else if (op_name == "or") {
      output.fill_(c10::Scalar(self.item<int64_t>() | other.item<int64_t>()));
    } else if (op_name == "xor") {
      output.fill_(c10::Scalar(self.item<int64_t>() ^ other.item<int64_t>()));
    } else {
      TORCH_CHECK(false, "Unknown operation to be performed over scalars ", op_name);
    }
  } else if (is_other_scalar) {
    handle_tensor_scalar_binary_op(self.contiguous(), other.item(), output, fmt::format("bitwise_{}_scalar", op_name));
  } else if (is_self_scalar) {
    handle_tensor_scalar_binary_op(other.contiguous(), self.item(), output, fmt::format("bitwise_{}_scalar", op_name));
  } else {
    handle_tensor_tensor_binary_op(self.expand(output_size).contiguous(),
                                   other.expand(output_size).contiguous(),
                                   output,
                                   fmt::format("bitwise_{}_tensor", op_name));
  }
  if (needs_output_copy) {
      output_.copy_(output);
  }
  return output_;
}

at::Tensor& bitwise_and_out_mps (const at::Tensor& self, const at::Tensor& other, at::Tensor& output) {
 return _bitwise_op_out_mps(self, other, output, "and");
}

at::Tensor& bitwise_or_out_mps (const at::Tensor& self, const at::Tensor& other, at::Tensor& output) {
 return _bitwise_op_out_mps(self, other, output, "or");
}

at::Tensor& bitwise_xor_out_mps (const at::Tensor& self, const at::Tensor& other, at::Tensor& output) {
 return _bitwise_op_out_mps(self, other, output, "xor");
}



TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("bitwise_and.Tensor_out", bitwise_and_out_mps);
  m.impl("bitwise_or.Tensor_out", bitwise_or_out_mps);
  m.impl("bitwise_xor.Tensor_out", bitwise_xor_out_mps);
}

} // anonymous namespace
