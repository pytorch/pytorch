#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Persistent.h>
#include <torch/custom_class.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

class Context final : public torch::jit::CustomClassHolder {
 public:
  Context(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& dilation,
      const int64_t groups,
      const c10::optional<Scalar> output_min,
      const c10::optional<Scalar> output_max);
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = default;
  Context& operator=(Context&&) = default;
  ~Context() = default;

  using State = std::tuple<
      Tensor,
      c10::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t,
      c10::optional<Scalar>,
      c10::optional<Scalar>>;

  Tensor run(const Tensor& input);
  State unpack() const;

 private:
  struct {
    Persistent::Image weight;
    Persistent::Buffer bias;
  } packed_;

  struct {
    Tensor weight;
    c10::optional<Tensor> bias;
  } original_;

  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  c10::optional<Scalar> output_min_;
  c10::optional<Scalar> output_max_;
};

Persistent::Image prepack_weights(const Tensor& weight) {
  Persistent::Image image = persistent()->image(
      VkExtent3D{},
      weight.options());

  return image;
}

Persistent::Buffer prepack_biases(const Tensor& bias) {
  Persistent::Buffer buffer = persistent()->buffer(
      bias.sizes(),
      bias.options());

  return buffer;
}

Context::Context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar> output_min,
    const c10::optional<Scalar> output_max)
  : packed_{
      prepack_weights(weight.contiguous()),
      prepack_biases(*bias),  // TODO (Ashkan)!
    },
    original_ {
      std::move(weight),
      std::move(bias),
    },
    stride_(std::move(stride)),
    padding_(std::move(padding)),
    dilation_(std::move(dilation)),
    groups_(groups),
    output_min_(output_min),
    output_max_(output_max) {
}

Tensor Context::run(const Tensor& input) {
  return Tensor{};
}

Context::State Context::unpack() const {
  return Context::State{
    original_.weight,
    original_.bias,
    stride_,
    padding_,
    dilation_,
    groups_,
    output_min_,
    output_max_,
  };
}

c10::intrusive_ptr<Context> conv2_clamp_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar> output_min,
    const c10::optional<Scalar> output_max) {
  return c10::make_intrusive<Context>(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      groups,
      output_min,
      output_min);
}

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<Context>& context) {
  return context->run(input);
}

typedef Context Conv2dOpContext;

TORCH_LIBRARY(vulkan, m) {
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Conv2dOpContext>& context) {
            return context->unpack();
          },
          // __setstate__
          [](Conv2dOpContext::State state) {
            return conv2_clamp_prepack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)));
          });
}

TORCH_LIBRARY(vulkan_prepack, m) {
  m.def(
      "conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dOpContext");
  m.def(
      "conv2d_clamp_run(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y");
}

TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl("conv2d_clamp_prepack", TORCH_FN(conv2_clamp_prepack));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl("conv2d_clamp_run", conv2d_clamp_run);
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
