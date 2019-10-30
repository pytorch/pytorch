#include <torch/nn/modules/activation.h>
#include <torch/nn/functional/activation.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

ELUImpl::ELUImpl(const ELUOptions& options_) : options(options_) {}

Tensor ELUImpl::forward(Tensor input) {
  return F::elu(input, options);
}

void ELUImpl::reset() {}

void ELUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ELU(alpha=" << options.alpha();
  if (options.inplace()) {
    stream << std::boolalpha  << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

SELUImpl::SELUImpl(const SELUOptions& options_) : options(options_) {}

Tensor SELUImpl::forward(Tensor input) {
  return F::selu(input, options);
}

void SELUImpl::reset() {}

void SELUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SELU(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

HardshrinkImpl::HardshrinkImpl(const HardshrinkOptions& options_)
    : options(options_) {}

Tensor HardshrinkImpl::forward(const Tensor& input) {
  return F::hardshrink(input, options);
}

void HardshrinkImpl::reset() {}

void HardshrinkImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Hardshrink(" << options.lambda() << ")";
}

// ============================================================================

HardtanhImpl::HardtanhImpl(const HardtanhOptions& options_)
    : options(options_) {
  reset();
}

Tensor HardtanhImpl::forward(Tensor input) {
  return F::hardtanh(input, options);
}

void HardtanhImpl::reset() {
  TORCH_CHECK(options.max_val() > options.min_val(),
              "max_val must be greater than min_val");
}

void HardtanhImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Hardtanh(min_val=" << options.min_val()
         << ", max_val=" << options.max_val();
  if (options.inplace()) {
    stream << std::boolalpha  << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

LeakyReLUImpl::LeakyReLUImpl(const LeakyReLUOptions& options_)
    : options(options_) {}

Tensor LeakyReLUImpl::forward(Tensor input) {
  return F::leaky_relu(input, options);
}

void LeakyReLUImpl::reset() {}

void LeakyReLUImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::LeakyReLU(negative_slope=" << options.negative_slope();
  if (options.inplace()) {
    stream << std::boolalpha  << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

Tensor LogSigmoidImpl::forward(const Tensor& input) {
  return F::logsigmoid(input);
}

void LogSigmoidImpl::reset() {}

void LogSigmoidImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::LogSigmoid()";
}

// ============================================================================

SoftmaxImpl::SoftmaxImpl(const SoftmaxOptions& options_)
    : options(options_) {}

void SoftmaxImpl::reset() {}

void SoftmaxImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softmax(dim=" << options.dim() << ")";
}

Tensor SoftmaxImpl::forward(const Tensor& input) {
  return F::softmax(input, options);
}

// ============================================================================

SoftminImpl::SoftminImpl(const SoftminOptions& options_)
    : options(options_) {}

void SoftminImpl::reset() {}

void SoftminImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softmin(dim=" << options.dim() << ")";
}

Tensor SoftminImpl::forward(const Tensor& input) {
  return F::softmin(input, options);
}

// ============================================================================

LogSoftmaxImpl::LogSoftmaxImpl(const LogSoftmaxOptions& options_)
    : options(options_) {}

void LogSoftmaxImpl::reset() {}

void LogSoftmaxImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::LogSoftmax(dim=" << options.dim() << ")";
}

Tensor LogSoftmaxImpl::forward(const Tensor& input) {
  return F::log_softmax(input, options);
}

// ============================================================================

void Softmax2dImpl::reset() {}

void Softmax2dImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softmax2d()";
}

Tensor Softmax2dImpl::forward(const Tensor& input) {
  TORCH_CHECK(input.dim() == 4, "Softmax2d requires a 4D tensor as input");
  return F::softmax(input, SoftmaxOptions(/*dim=*/1));
}

// ============================================================================

PReLUImpl::PReLUImpl(const PReLUOptions& options_) : options(options_) {
  reset();
}

Tensor PReLUImpl::forward(const Tensor& input) {
  return F::prelu(input, weight);
}

void PReLUImpl::reset() {
  weight = register_parameter("weight",
    torch::full(options.num_parameters(), options.init()));
}

void PReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PReLU(num_parameters="
         << options.num_parameters() << ")";
}

// ============================================================================

ReLUImpl::ReLUImpl(const ReLUOptions& options_) : options(options_) {}

Tensor ReLUImpl::forward(Tensor input) {
  return F::relu(input, options);
}

void ReLUImpl::reset() {}

void ReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReLU(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

ReLU6Impl::ReLU6Impl(const ReLU6Options& options_) : options(options_) {}

Tensor ReLU6Impl::forward(Tensor input) {
  return F::relu6(input, options);
}

void ReLU6Impl::reset() {}

void ReLU6Impl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReLU6(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

RReLUImpl::RReLUImpl(const RReLUOptions& options_) : options(options_) {}

Tensor RReLUImpl::forward(Tensor input) {
  return F::rrelu(input, options, is_training());
}

void RReLUImpl::reset() {}

void RReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::RReLU(lower=" << options.lower()
         << ", upper=" << options.upper();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

CELUImpl::CELUImpl(const CELUOptions& options_) : options(options_) {}

Tensor CELUImpl::forward(Tensor input) {
  return F::celu(input, options);
}

void CELUImpl::reset() {}

void CELUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CELU(alpha=" << options.alpha();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

Tensor SigmoidImpl::forward(const Tensor& input) {
  return torch::sigmoid(input);
}

void SigmoidImpl::reset() {}

void SigmoidImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Sigmoid()";
}

// ============================================================================

SoftplusImpl::SoftplusImpl(const SoftplusOptions& options_)
  : options(options_) {}

Tensor SoftplusImpl::forward(const Tensor& input) {
  return F::softplus(input, options);
}

void SoftplusImpl::reset() {}

void SoftplusImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softplus(beta=" << options.beta()
         << ", threshold=" << options.threshold() << ")";
}

// ============================================================================

SoftshrinkImpl::SoftshrinkImpl(const SoftshrinkOptions& options_)
    : options(options_) {}

Tensor SoftshrinkImpl::forward(const Tensor& input) {
  return F::softshrink(input, options);
}

void SoftshrinkImpl::reset() {}

void SoftshrinkImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softshrink(" << options.lambda() << ")";
}

// ============================================================================

Tensor SoftsignImpl::forward(const Tensor& input) {
  return F::softsign(input);
}

void SoftsignImpl::reset() {}

void SoftsignImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softsign()";
}

// ============================================================================

Tensor TanhImpl::forward(const Tensor& input) {
  return torch::tanh(input);
}

void TanhImpl::reset() {}

void TanhImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Tanh()";
}

// ============================================================================

Tensor TanhshrinkImpl::forward(const Tensor& input) {
  return F::tanhshrink(input);
}

void TanhshrinkImpl::reset() {}

void TanhshrinkImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Tanhshrink()";
}

// ============================================================================

ThresholdImpl::ThresholdImpl(const ThresholdOptions& options_)
    : options(options_) {}

Tensor ThresholdImpl::forward(Tensor input) {
  return F::threshold(input, options);
}

void ThresholdImpl::reset() {}

void ThresholdImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Threshold(threshold=" << options.threshold()
         << ", value=" << options.value();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

} // namespace nn
} // namespace torch
