#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/bitwise_and_native.h>
#include <ATen/ops/bitwise_or_native.h>
#include <ATen/ops/bitwise_xor_native.h>
#include <ATen/ops/logical_not_native.h>
#include <fmt/format.h>

namespace at::native {
namespace mps {
static MetalShaderLibrary lib(R"METAL(

kernel void bitwise_and_tensor_tensor(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  *b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = a[offset] & b [offset];
}}

kernel void bitwise_and_tensor_scalar(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  &b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = a[offset] & b;
}}


kernel void bitwise_or_tensor_tensor(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  *b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = a[offset] | b [offset];
}}

kernel void bitwise_or_tensor_scalar(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  &b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = a[offset] | b;
}}

kernel void bitwise_xor_tensor_tensor(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  *b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = a[offset] ^ b [offset];
}}

kernel void bitwise_xor_tensor_scalar(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  &b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = a[offset] ^ b;
}}

kernel void bitwise_lshift_tensor_tensor(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  *b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = static_cast<{0}>(a[offset]) << b [offset];
}}

kernel void bitwise_lshift_tensor_scalar(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  &b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = static_cast<{0}>(a[offset]) << b;
}}

kernel void bitwise_lshift_scalar_tensor(device {0}  *out [[buffer(0)]],
                         constant {1}  &a [[buffer(1)]],
                         constant {2}  *b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = static_cast<{0}>(a) << b[offset];
}}

kernel void bitwise_rshift_tensor_tensor(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  *b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = static_cast<{0}>(a[offset]) >> b [offset];
}}

kernel void bitwise_rshift_tensor_scalar(device {0}  *out [[buffer(0)]],
                         constant {1}  *a [[buffer(1)]],
                         constant {2}  &b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = static_cast<{0}>(a[offset]) >> b;
}}

kernel void bitwise_rshift_scalar_tensor(device {0}  *out [[buffer(0)]],
                         constant {1}  &a [[buffer(1)]],
                         constant {2}  *b [[buffer(2)]],
                         uint offset [[thread_position_in_grid]]) {{
  out[offset] = static_cast<{0}>(a) >> b[offset];
}}

)METAL",
                              3);

static inline std::string getMetalType(const c10::ScalarType scalar_type) {
  TORCH_CHECK(c10::isIntegralType(scalar_type, /*includesBool=*/true), "Unsupported type");
  return scalarToMetalTypeString(scalar_type);
}

static inline std::string getMetalType(const Tensor& t) {
  return getMetalType(t.scalar_type());
}

static id<MTLComputePipelineState> getCPLState(const Tensor& t1,
                                               const Tensor& t2,
                                               const Tensor& t3,
                                               const std::string& fname) {
  return lib.getPipelineStateForFunc(fname, {getMetalType(t1), getMetalType(t2), getMetalType(t3)});
}

static void handle_binary_op(const Tensor& self, const Tensor& other, Tensor& output, const std::string& kernel_name) {
  using namespace at::mps;
  MPSStream* stream = getCurrentMPSStream();
  auto cplState = getCPLState(output, self, other, kernel_name);
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }

  dispatch_sync(stream->queue(), ^() {
    // this function call is a no-op if MPS Profiler is not enabled
    getMPSProfiler().beginProfileKernel(cplState, kernel_name, {self, other});

    id<MTLComputeCommandEncoder> commandEncoder = stream->commandEncoder();

    [commandEncoder pushDebugGroup:[NSString stringWithFormat:@"Dispatch %s kernel", kernel_name.c_str()]];
    [commandEncoder setComputePipelineState:cplState];
    mtl_setArgs(commandEncoder, output, self, other);
    mtl_dispatch1DJob(commandEncoder, cplState, length);

    getMPSProfiler().endProfileKernel(cplState);
  });
}

static void _bitwise_op_out_mps(const Tensor& self,
                                const Tensor& other,
                                const Tensor& output_,
                                const std::string& op_name,
                                bool is_commutative = true) {
  using namespace at::mps;
  const bool is_self_scalar = self.dim() == 0;
  const bool is_other_scalar = other.dim() == 0;

  Tensor output = output_;
  bool needs_output_copy = false;

  auto output_size = at::infer_size_dimvector(self.sizes(), other.sizes());
  resize_output(output, output_size);
  if (!output.is_contiguous()) {
    output = output.contiguous();
    needs_output_copy = true;
  }
  if (is_other_scalar && is_self_scalar) {
    if (op_name == "and") {
      output.fill_(c10::Scalar(self.item<int64_t>() & other.item<int64_t>()));
    } else if (op_name == "or") {
      output.fill_(c10::Scalar(self.item<int64_t>() | other.item<int64_t>()));
    } else if (op_name == "xor") {
      output.fill_(c10::Scalar(self.item<int64_t>() ^ other.item<int64_t>()));
    } else if (op_name == "lshift") {
      output.fill_(c10::Scalar(self.item<int64_t>() << other.item<int64_t>()));
    } else if (op_name == "rshift") {
      output.fill_(c10::Scalar(self.item<int64_t>() >> other.item<int64_t>()));
    } else {
      TORCH_CHECK(false, "Unknown operation to be performed over scalars ", op_name);
    }
  } else if (is_other_scalar) {
    handle_binary_op(self.contiguous(), other, output, fmt::format("bitwise_{}_tensor_scalar", op_name));
  } else if (is_self_scalar) {
    if (!is_commutative) {
      handle_binary_op(self, other.contiguous(), output, fmt::format("bitwise_{}_scalar_tensor", op_name));
    } else {
      handle_binary_op(other.contiguous(), self, output, fmt::format("bitwise_{}_tensor_scalar", op_name));
    }
  } else {
    handle_binary_op(self.expand(output_size).contiguous(),
                     other.expand(output_size).contiguous(),
                     output,
                     fmt::format("bitwise_{}_tensor_tensor", op_name));
  }
  if (needs_output_copy) {
    output_.copy_(output);
  }
  return;
}

} // namespace mps
namespace {
void lshift_kernel_mps(TensorIteratorBase& iter) {
  mps::_bitwise_op_out_mps(iter.input(0), iter.input(1), iter.output(0), "lshift", false);
}

void rshift_kernel_mps(TensorIteratorBase& iter) {
  mps::_bitwise_op_out_mps(iter.input(0), iter.input(1), iter.output(0), "rshift", false);
}

} // namespace

TORCH_IMPL_FUNC(bitwise_and_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::_bitwise_op_out_mps(self, other, output, "and");
}

TORCH_IMPL_FUNC(bitwise_or_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::_bitwise_op_out_mps(self, other, output, "or");
}

TORCH_IMPL_FUNC(bitwise_xor_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::_bitwise_op_out_mps(self, other, output, "xor");
}

REGISTER_MPS_DISPATCH(lshift_stub, &lshift_kernel_mps)
REGISTER_MPS_DISPATCH(rshift_stub, &rshift_kernel_mps)

} // namespace at::native
