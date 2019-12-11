#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/functional/activation.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies elu over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ELU to learn
/// about the exact behavior of this module.
class TORCH_API ELUImpl : public torch::nn::Cloneable<ELUImpl> {
 public:
  explicit ELUImpl(const ELUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `ELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ELUOptions options;
};

TORCH_MODULE(ELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the selu function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.SELU to learn
/// about the exact behavior of this module.
class TORCH_API SELUImpl : public torch::nn::Cloneable<SELUImpl> {
 public:
  explicit SELUImpl(const SELUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `SELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  SELUOptions options;
};

TORCH_MODULE(SELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the hard shrinkage function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Hardshrink to learn
/// about the exact behavior of this module.
class TORCH_API HardshrinkImpl : public torch::nn::Cloneable<HardshrinkImpl> {
 public:
  explicit HardshrinkImpl(const HardshrinkOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Hardshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  HardshrinkOptions options;
};

TORCH_MODULE(Hardshrink);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardtanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the HardTanh function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Hardtanh to learn
/// about the exact behavior of this module.
class TORCH_API HardtanhImpl : public torch::nn::Cloneable<HardtanhImpl> {
 public:
  explicit HardtanhImpl(const HardtanhOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `Hardtanh` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  HardtanhOptions options;
};

TORCH_MODULE(Hardtanh);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LeakyReLU function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LeakyReLU to learn
/// about the exact behavior of this module.
class TORCH_API LeakyReLUImpl : public torch::nn::Cloneable<LeakyReLUImpl> {
 public:
  explicit LeakyReLUImpl(const LeakyReLUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `LeakyReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  LeakyReLUOptions options;
};

TORCH_MODULE(LeakyReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LogSigmoid function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LogSigmoid to learn
/// about the exact behavior of this module.
class TORCH_API LogSigmoidImpl : public torch::nn::Cloneable<LogSigmoidImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `LogSigmoid` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(LogSigmoid);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Softmax function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Softmax to learn
/// about the exact behavior of this module.
class TORCH_API SoftmaxImpl : public torch::nn::Cloneable<SoftmaxImpl> {
 public:
  explicit SoftmaxImpl(int64_t dim) : SoftmaxImpl(SoftmaxOptions(dim)) {}
  explicit SoftmaxImpl(const SoftmaxOptions& options_);

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softmax` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  SoftmaxOptions options;
};

TORCH_MODULE(Softmax);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Softmin function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Softmin to learn
/// about the exact behavior of this module.
class TORCH_API SoftminImpl : public torch::nn::Cloneable<SoftminImpl> {
 public:
  explicit SoftminImpl(int64_t dim) : SoftminImpl(SoftminOptions(dim)) {}
  explicit SoftminImpl(const SoftminOptions& options_);
  
  Tensor forward(const Tensor& input);

  void reset() override;
  
  /// Pretty prints the `Softmin` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  SoftminOptions options;
};

TORCH_MODULE(Softmin);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSoftmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LogSoftmax function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LogSoftmax to learn
/// about the exact behavior of this module.
class TORCH_API LogSoftmaxImpl : public torch::nn::Cloneable<LogSoftmaxImpl> {
 public:
  explicit LogSoftmaxImpl(int64_t dim) : LogSoftmaxImpl(LogSoftmaxOptions(dim)) {}
  explicit LogSoftmaxImpl(const LogSoftmaxOptions& options_);

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `LogSoftmax` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  LogSoftmaxOptions options;
};

TORCH_MODULE(LogSoftmax);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Softmax2d function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Softmax2d to learn
/// about the exact behavior of this module.
class TORCH_API Softmax2dImpl : public torch::nn::Cloneable<Softmax2dImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softmax2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(Softmax2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the PReLU function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.PReLU to learn
/// about the exact behavior of this module.
class TORCH_API PReLUImpl : public torch::nn::Cloneable<PReLUImpl> {
 public:
  explicit PReLUImpl(const PReLUOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `PReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  PReLUOptions options;

  /// The learned weight.
  Tensor weight;
};

TORCH_MODULE(PReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ReLU function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReLU to learn
/// about the exact behavior of this module.
class TORCH_API ReLUImpl : public torch::nn::Cloneable<ReLUImpl> {
 public:
  explicit ReLUImpl(const ReLUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `ReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReLUOptions options;
};

TORCH_MODULE(ReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU6 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ReLU6 function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReLU6 to learn
/// about the exact behavior of this module.
class TORCH_API ReLU6Impl : public torch::nn::Cloneable<ReLU6Impl> {
 public:
  explicit ReLU6Impl(const ReLU6Options& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `ReLU6` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReLU6Options options;
};

TORCH_MODULE(ReLU6);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the RReLU function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.RReLU to learn
/// about the exact behavior of this module.
class TORCH_API RReLUImpl : public torch::nn::Cloneable<RReLUImpl> {
 public:
  explicit RReLUImpl(const RReLUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `RReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  RReLUOptions options;
};

TORCH_MODULE(RReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies celu over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.CELU to learn
/// about the exact behavior of this module.
class TORCH_API CELUImpl : public torch::nn::Cloneable<CELUImpl> {
 public:
  explicit CELUImpl(const CELUOptions& options_ = {});

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `CELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  CELUOptions options;
};

TORCH_MODULE(CELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies glu over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.GLU to learn
/// about the exact behavior of this module.
class TORCH_API GLUImpl : public torch::nn::Cloneable<GLUImpl> {
 public:
  explicit GLUImpl(const GLUOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `GLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  GLUOptions options;
};

TORCH_MODULE(GLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies gelu over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.GELU to learn
/// about the exact behavior of this module.
class TORCH_API GELUImpl : public torch::nn::Cloneable<GELUImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `GELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(GELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies sigmoid over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Sigmoid to learn
/// about the exact behavior of this module.
class TORCH_API SigmoidImpl : public torch::nn::Cloneable<SigmoidImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Sigmoid` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(Sigmoid);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softplus ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies softplus over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Softplus to learn
/// about the exact behavior of this module.
class TORCH_API SoftplusImpl : public torch::nn::Cloneable<SoftplusImpl> {
 public:
  explicit SoftplusImpl(const SoftplusOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softplus` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  SoftplusOptions options;
};

TORCH_MODULE(Softplus);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the soft shrinkage function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Softshrink to learn
/// about the exact behavior of this module.
class TORCH_API SoftshrinkImpl : public torch::nn::Cloneable<SoftshrinkImpl> {
 public:
  explicit SoftshrinkImpl(const SoftshrinkOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  SoftshrinkOptions options;
};

TORCH_MODULE(Softshrink);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softsign ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Softsign over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Softsign to learn
/// about the exact behavior of this module.
class TORCH_API SoftsignImpl : public torch::nn::Cloneable<SoftsignImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softsign` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(Softsign);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Tanh over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Tanh to learn
/// about the exact behavior of this module.
class TORCH_API TanhImpl : public torch::nn::Cloneable<TanhImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Tanh` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(Tanh);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tanhshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Tanhshrink over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Tanhshrink to learn
/// about the exact behavior of this module.
class TORCH_API TanhshrinkImpl : public torch::nn::Cloneable<TanhshrinkImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Tanhshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(Tanhshrink);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Threshold ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Threshold function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Threshold to learn
/// about the exact behavior of this module.
class TORCH_API ThresholdImpl : public torch::nn::Cloneable<ThresholdImpl> {
 public:
  ThresholdImpl(double threshold, double value)
    : ThresholdImpl(ThresholdOptions(threshold, value)) {}
  explicit ThresholdImpl(const ThresholdOptions& options_);

  Tensor forward(Tensor input);

  void reset() override;

  /// Pretty prints the `Threshold` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ThresholdOptions options;
};

TORCH_MODULE(Threshold);

} // namespace nn
} // namespace torch
