#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/Cross.h>

namespace at {
namespace meta {

TORCH_META_FUNC(cross)
(const Tensor & input, const Tensor & other, const c10::optional<int64_t> dimension) {
  const Tensor& out = maybe_get_output(0);

  auto device1 = input.device().type();
  auto device2 = other.device().type();

  if (out.defined()) {
    auto device_res = out.device().type();
    TORCH_CHECK(device_res == device1, "out and input must have the same device type. out: ", device_res, " input: ", device1);
    TORCH_CHECK(!out.is_cuda() || out.get_device() == input.get_device(),
                "device of out (", input.get_device(), ") must match device of input (", other.get_device(), ")");
  }

  TORCH_CHECK(device1 == device2, "input and other must have the same device type. input: ", device1, " other: ", device2);
  TORCH_CHECK(!input.is_cuda() || input.get_device() == other.get_device(),
              "device of input (", input.get_device(), ") must match device of other (", other.get_device(), ")");
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
}

} // namespace at::meta

namespace native {

DEFINE_DISPATCH(cross_stub);

TORCH_IMPL_FUNC(cross_out)
(const Tensor & input, const Tensor & other, const c10::optional<int64_t> dimension, const Tensor & out) {
  auto device1 = input.device().type();

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
}

}} // at::native
