#import <Foundation/Foundation.h>

#include <ATen/Tensor.h>
#include <torch/custom_class.h>

namespace at {
namespace native {
namespace metal {

using SerializationTypeConv2dPrePack = std::tuple<
    Tensor,
    c10::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    c10::optional<Scalar>,
    c10::optional<Scalar>>;

class Conv2dOpContext : public torch::jit::CustomClassHolder {
 public:
  SerializationTypeConv2dPrePack pack() {
    return std::make_tuple(
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        output_min,
        output_max);
  }
  Conv2dOpContext() = delete;
  Conv2dOpContext(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      const std::vector<int64_t>& stride,
      const std::vector<int64_t>& padding,
      const std::vector<int64_t>& dilation,
      int64_t groups,
      c10::optional<Scalar> output_min,
      c10::optional<Scalar> output_max)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        stride(stride),
        padding(padding),
        dilation(dilation),
        groups(groups),
        output_min(output_min),
        output_max(output_max) {}

  Tensor weight;
  c10::optional<Tensor> bias;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  int64_t groups;
  c10::optional<Scalar> output_min;
  c10::optional<Scalar> output_max;
  id extra = nil;
};

c10::intrusive_ptr<Conv2dOpContext> unpack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    c10::optional<Scalar> output_min,
    c10::optional<Scalar> output_max);

c10::intrusive_ptr<Conv2dOpContext> conv2d_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    c10::optional<Scalar> output_min,
    c10::optional<Scalar> output_max);

Tensor conv2d_prepack_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& op_context);

Tensor copy_to_host(const Tensor& input);

} // namespace metal
} // namespace native
} // namespace at
