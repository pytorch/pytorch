#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight, const c10::optional<Tensor>& bias_opt) {
  TORCH_CHECK(false, "mkldnn_linear: ATen not compiled with MKLDNN support");
}
Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_linear_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_linear_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output_t,
    const Tensor& weight, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "mkldnn_linear_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

namespace {

static inline bool is_input_dtype_valid(const Tensor& t) {
  auto st = t.scalar_type();
  return (t.is_mkldnn() && (st == ScalarType::Float || st == ScalarType::BFloat16)) ||
      (!t.is_mkldnn() && st == ScalarType::BFloat16);
}

// reshape input/output in 2d view, might take a memcpy
static inline Tensor view_2d(const Tensor& t) {
  return t.dim() == 2 ? t : t.reshape({-1, t.size(t.dim() - 1)});
}

static inline Tensor _contiguous(const Tensor& t) {
  return t.is_mkldnn() ? t : t.contiguous();
}

} // anonymous namespace

// Note on mkldnn_linear layout and dtype propagation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// a) mkldnn_linear (e.g. mkldnn privimitive inner_product) will be used for
//   * mkldnn layout (fp32 and bf16)
//   * dense layout (bf16)
//
// b) dense layout (fp32) will still call `mkl` sgemm, reason that we skip mkl
//    for dense layout (bf16) is that `cblas_gemm_bf16bf16f32` will have C matrix
//    in fp32 which is against pytorch dtype propagation rule. And doing an post
//    fp32->bf16 conversion is adverse for performance.
//
// c) mkldnn may choose blocked memory format for weight due to performance reasons.
//    On mkldnn layout, weight can be prepacked to blocked to save reorder overhead
//    for inference. Weight is always plain format (OI) for training.
//
// 1. mkldnn layout propagation
//   (input, output are mkldnn layout)
// +---------------------------------------------------------------------------+
// | propagation          | input     | weight    | output    || weight layout |
// |----------------------+-----------+-----------+-----------++---------------|
// | inference prepacked  | fp32|bf16 | fp32|bf16 | fp32|bf16 || mkldnn        |
// | training forward     | fp32|bf16 | fp32      | fp32|bf16 || dense         |
// | training backward    | fp32|bf16 | fp32      | fp32|bf16 || dense         |
// +---------------------------------------------------------------------------+
//
// 2. dense layout propagation
//   (input, output are dense layout)
// +---------------------------------------------------------------------------+
// | propagation          | input     | weight    | output    || weight layout |
// |----------------------+-----------+-----------+-----------++---------------|
// | inference prepacked  | N/A       | N/A       | N/A       || N/A           |
// | training forward     | bf16      | bf16      | bf16      || dense         |
// | training backward    | bf16      | bf16      | bf16      || dense         |
// +---------------------------------------------------------------------------+
//

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight_t, const c10::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  const int64_t dim = self.dim();
  TORCH_CHECK(self.dim() != 0,
      "mkldnn_linear: input needs to has dim at least 1, input dim ",
      self.dim());
  TORCH_CHECK(is_input_dtype_valid(self),
      "mkldnn_linear: input needs to be mkldnn layout in float32|bfloat16 or dense layout in bfloat16");
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_linear: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  const ideep::tensor x = itensor_from_tensor(_contiguous(view_2d(self)));
  const ideep::tensor w = itensor_from_tensor(_contiguous(weight_t));

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight_t.size(0));

  auto output = at::empty({0}, self.options());
  ideep::tensor y;
  if (!self.is_mkldnn()) {
    output.resize_(output_size);
    y = itensor_from_tensor(view_2d(output));
  }

  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias);
    ideep::inner_product_forward::compute(x, w, b, y);
  } else {
    ideep::inner_product_forward::compute(x, w, y);
  }

  // dense layout (bfloat16)
  if (!self.is_mkldnn()) {
    return output;
  }
  // mkldnn layout (float and bfloat16)
  if (self.dim() != 2) {
    return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                   self.options().device_opt()).reshape(output_size);
  }
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight){

  auto grad_output_reshaped = view_2d(grad_output);
  const ideep::tensor grady = itensor_from_tensor(grad_output_reshaped);
  // weight is always dense tensor for training.
  const ideep::tensor w = itensor_view_from_dense(weight);

  auto grad_input = at::empty({0}, grad_output.options());
  ideep::tensor gradx;
  if (!grad_output.is_mkldnn()) {
    grad_input.resize_(input_size);
    gradx = itensor_from_tensor(view_2d(grad_input));
  }
  ideep::inner_product_backward_data::compute(
    grady, w, {grad_output_reshaped.size(0), weight.size(1)}, gradx);

  // dense layout (bfloat16)
  if (!grad_output.is_mkldnn()) {
    return grad_input;
  }
  // mkldnn layout (float and bfloat16)
  if (input_size.size() > 2) {
    return new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                   grad_output.options().device_opt()).reshape(input_size);
  }
  return new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {

  const ideep::tensor grady = itensor_from_tensor(view_2d(grad_output));
  const ideep::tensor x = itensor_from_tensor(view_2d(input));
  ideep::tensor gradw, gradb;
  if (bias_defined) {
    ideep::inner_product_backward_weights::compute(x, grady, gradw, gradb);
  } else {
    ideep::inner_product_backward_weights::compute(x, grady, gradw);
  }

  return std::tuple<Tensor, Tensor>{
    mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt())),
    mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradb),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt()))};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, std::array<bool,3> output_mask) {
  if (input.is_mkldnn()) {
    TORCH_CHECK(grad_output.is_mkldnn(),
        "mkldnn_linear_backward (mkldnn layout input): grad_output need to be mkldnn layout");
    TORCH_CHECK(input.scalar_type() == grad_output.scalar_type() &&
        (input.scalar_type() == ScalarType::Float || input.scalar_type() == ScalarType::BFloat16),
        "mkldnn_linear_backward (mkldnn layout input): ",
        "grad_output and input need to be the same dtype (float32 or bfloat16)");
    TORCH_CHECK(weight.layout() == Layout::Strided && weight.scalar_type() == ScalarType::Float,
        "mkldnn_linear_backward (mkldnn layout input): weight needs to be dense layout in float32");
  } else {
    TORCH_CHECK(grad_output.layout() == Layout::Strided,
        "mkldnn_linear_backward (dense layout input): grad_output need to be dense layout");
    TORCH_CHECK(input.scalar_type() == grad_output.scalar_type() && (input.scalar_type() == ScalarType::BFloat16),
        "mkldnn_linear_backward (dense layout input): grad_output and input need to be bfloat16");
    TORCH_CHECK(weight.layout() == Layout::Strided && weight.scalar_type() == ScalarType::BFloat16,
        "mkldnn_linear_backward (dense layout input): weight needs to be dense layout in bfloat16");
  }

  Tensor input_t = _contiguous(input);
  Tensor grad_output_t = _contiguous(grad_output);
  // weight is always dense tensor for training.
  Tensor weight_t = weight.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_linear_backward_input(input.sizes(), grad_output_t, weight_t);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_linear_backward_weights(grad_output_t, input_t, weight_t, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}} // at::native

#endif // AT_MKLDNN_EBABLED
