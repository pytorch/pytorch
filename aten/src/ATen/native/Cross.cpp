#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/Cross.h>

namespace at {
namespace meta {

DEFINE_DISPATCH(cross_stub);

TORCH_META_FUNC(cross)
(const Tensor & input, const Tensor & other, const c10::optional<int64_t> dimension) {
  const Tensor& out = maybe_get_output(0);
  
  auto device_res = out.device().type();
  TORCH_CHECK(device_res == kCPU || device_res == kCUDA, "cross only supports CPU and CUDA devices, out got: ", device_res);
  auto device1 = input.device().type();
  TORCH_CHECK(device1 == kCPU || device1 == kCUDA, "cross only supports CPU and CUDA devices, input got: ", device1);
  auto device2 = other.device().type();
  TORCH_CHECK(device2 == kCPU || device2 == kCUDA, "cross only supports CPU and CUDA devices, other got: ", device2);
  TORCH_CHECK(device_res == device1, "out and input must have the same device type. out: ", device_res, " input: ", device1);
  TORCH_CHECK(device1 == device2, "input and other must have the same device type. input: ", device1, " other: ", device2);
  TORCH_CHECK(!out.is_cuda() || out.get_device() == input.get_device(), "device of out (", input.get_device(), ") must match device of input (", other.get_device(), ")");
  TORCH_CHECK(!input.is_cuda() || input.get_device() == other.get_device(), "device of input (", input.get_device(), ") must match device of other (", other.get_device(), ")");
  TORCH_CHECK(input.dim() == other.dim(), "inconsistent tensors dimensions input: ", input.dim(), " other: ", other.dim());
  TORCH_CHECK(input.sizes() == other.sizes(), "inconsistent tensors sizes input: ", input.sizes(), " other: ", other.sizes());

  int64_t dim = -1;
  if(!dimension.has_value()) {
    for(int64_t i = 0; i < input.dim(); i++) {
      if(input.size(i) == 3) {
        dim = i;
        break;
      }
    }
    TORCH_CHECK(dim >= 0, "no dimension of size 3 in input");
  } else {
    dim = maybe_wrap_dim(dimension.value(), input.dim());
    TORCH_CHECK(input.size(dim) == 3, "dimension ", dimension.value(), " does not have size 3");
  }

  set_output(input.sizes(), input.options());

  return out;
}

TORCH_IMPL_FUNC(gather_out)
(const Tensor & input, const Tensor & other, const c10::optional<int64_t> dimension, const Tensor & out) {
  int64_t dim = -1;
  if(!dimension.has_value()) {
    for(int64_t i = 0; i < input.dim(); i++) {
      if(input.size(i) == 3) {
        dim = i;
        break;
      }
    }
  } else {
    dim = maybe_wrap_dim(dimension.value(), input.dim());
  }

  cross_stub(device1, out, input, other, dim);
  return out;
}

}} // namespace at::meta