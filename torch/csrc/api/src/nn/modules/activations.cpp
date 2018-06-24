#include <torch/nn/modules/activations.h>

#include <torch/nn/modules/functional.h>

#include <ATen/ATen.h>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Threshold ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ThresholdOptions::ThresholdOptions(double threshold, double value)
    : threshold_(threshold), value_(value) {}

ThresholdImpl::ThresholdImpl(ThresholdOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::threshold(
            input, this->options.threshold_, this->options.value_);
      }),
      options(options) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ReLUImpl::ReLUImpl() : ThresholdImpl({/*threshold=*/0, /*value=*/0}) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RReluOptions::RReluOptions(double lower, double upper)
    : lower_(lower), upper_(upper) {}

RReluImpl::RReluImpl(RReluOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::rrelu(
            input, this->options.lower_, this->options.upper_, is_training());
      }),
      options(options) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SigmoidImpl::SigmoidImpl() : FunctionalImpl(at::sigmoid) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TanhImpl::TanhImpl() : FunctionalImpl(at::tanh) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ELUOptions::ELUOptions(double alpha) : alpha_(alpha) {}

ELUImpl::ELUImpl(ELUOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::elu(input, this->options.alpha_);
      }),
      options(options) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SELUImpl::SELUImpl() : FunctionalImpl(at::selu) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GLUOptions::GLUOptions(int64_t dim) : dim_(dim) {}

GLUImpl::GLUImpl(GLUOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::glu(input, this->options.dim_);
      }),
      options(options) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HardshrinkOptions::HardshrinkOptions(double lambda) : lambda_(lambda) {}

HardshrinkImpl::HardshrinkImpl(HardshrinkOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::hardshrink(input, this->options.lambda_);
      }),
      options(options) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardtanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HardtanhOptions::HardtanhOptions(double min_val, double max_val)
    : min_val_(min_val), max_val_(max_val) {}

HardtanhImpl::HardtanhImpl(HardtanhOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::hardtanh(
            input, this->options.min_val_, this->options.max_val_);
      }),
      options(options) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LeakyReLUOptions::LeakyReLUOptions(double negative_slope)
    : negative_slope_(negative_slope) {}

LeakyReLUImpl::LeakyReLUImpl(LeakyReLUOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::leaky_relu(input, this->options.negative_slope_);
      }),
      options(options) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softplus ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SoftplusOptions::SoftplusOptions(double beta) : beta_(beta) {}

SoftplusImpl::SoftplusImpl(SoftplusOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::softplus(
            input, this->options.beta_, this->options.threshold_);
      }),
      options(options) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softsign ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SoftsignImpl::SoftsignImpl()
    : FunctionalImpl([](Variable input) { return input / (input.abs() + 1); }) {
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SoftmaxOptions::SoftmaxOptions(int64_t dim) : dim_(dim) {}

SoftmaxImpl::SoftmaxImpl(SoftmaxOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::softmax(input, this->options.dim_);
      }),
      options(options) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSoftmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LogSoftmaxOptions::LogSoftmaxOptions(int64_t dim) : dim_(dim) {}

LogSoftmaxImpl::LogSoftmaxImpl(LogSoftmaxOptions options)
    : FunctionalImpl([this](Variable input) {
        return at::log_softmax(input, this->options.dim_);
      }),
      options(options) {}

} // namespace nn
} // namespace torch
