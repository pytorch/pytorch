#include <vector>

#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/LinearPrepack.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/utils/ParamUtils.h>
#include <c10/util/irange.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace linear {

Tensor linear_eltwise_run(
    const Tensor& input,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    std::string attr,
    std::vector<c10::optional<at::Scalar>> scalars,
    c10::optional<std::string> algorithm) {
  auto input_size = input.sizes();

  const int64_t dim = input.dim();
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});

  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight_t.size(0));
  auto output = at::empty(output_size, input.options());

  if (dim != 2) {
    std::vector<int64_t> output_size_reshaped = {input_reshaped.size(0),
                                                 weight_t.size(0)};
    output = output.reshape(output_size_reshaped);
  }

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  ideep::tensor mkldnn_output = itensor_from_tensor(output);

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  const ideep::tensor mkldnn_input = itensor_view_from_dense(input_reshaped);

  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }
  const ideep::tensor w = itensor_from_tensor(weight_t);

  auto it = fx_fusion_attr_map().find(attr);
  TORCH_CHECK(it != fx_fusion_attr_map().end(), "Fusion behavior undefined.");
  ideep::attr_t op_attr = it->second(scalars, algorithm);

  if (mkldnn_bias.has_value()) {
    ideep::inner_product_forward::compute(
        mkldnn_input,
        w,
        mkldnn_bias.value(),
        mkldnn_output,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        op_attr);
  } else {
    ideep::inner_product_forward::compute(
        mkldnn_input,
        w,
        mkldnn_output,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        op_attr);
  }

  if (dim != 2) {
    output = output.reshape(output_size);
  }

  return output;
}

} // namespace linear
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
