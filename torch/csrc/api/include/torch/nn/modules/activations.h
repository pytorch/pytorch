#pragma once

#include <torch/nn/modules/functional.h>
#include <torch/nn/pimpl.h>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Threshold ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct ThresholdOptions {
  ThresholdOptions(double threshold, double value);
  TORCH_ARG(double, threshold);
  TORCH_ARG(double, value);
};

struct ThresholdImpl : FunctionalImpl {
  explicit ThresholdImpl(ThresholdOptions options);
  ThresholdOptions options;
};

TORCH_MODULE(Threshold);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct ReLUImpl : ThresholdImpl {
  ReLUImpl();
};

TORCH_MODULE(ReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct RReluOptions {
  RReluOptions(double lower, double upper);
  TORCH_ARG(double, lower) = 1.0 / 8;
  TORCH_ARG(double, upper) = 1.0 / 3;
};

struct RReluImpl : FunctionalImpl {
  explicit RReluImpl(RReluOptions options);
  RReluOptions options;
};

TORCH_MODULE(RRelu);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct SigmoidImpl : FunctionalImpl {
  SigmoidImpl();
};

TORCH_MODULE(Sigmoid);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct TanhImpl : FunctionalImpl {
  TanhImpl();
};

TORCH_MODULE(Tanh);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct ELUOptions {
  ELUOptions(double alpha);
  TORCH_ARG(double, alpha) = 1.0;
};

struct ELUImpl : FunctionalImpl {
  explicit ELUImpl(ELUOptions options);
  ELUOptions options;
};

TORCH_MODULE(ELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct SELUImpl : FunctionalImpl {
  SELUImpl();
};

TORCH_MODULE(SELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct GLUOptions {
  GLUOptions(int64_t dim);
  TORCH_ARG(int64_t, dim);
};

struct GLUImpl : FunctionalImpl {
  explicit GLUImpl(GLUOptions options);
  GLUOptions options;
};

TORCH_MODULE(GLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct HardshrinkOptions {
  HardshrinkOptions(double lambda);
  TORCH_ARG(double, lambda);
};

struct HardshrinkImpl : FunctionalImpl {
  explicit HardshrinkImpl(HardshrinkOptions options);
  HardshrinkOptions options;
};

TORCH_MODULE(Hardshrink);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardtanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct HardtanhOptions {
  HardtanhOptions(double min_val, double max_val);
  TORCH_ARG(double, min_val);
  TORCH_ARG(double, max_val);
};

struct HardtanhImpl : FunctionalImpl {
  explicit HardtanhImpl(HardtanhOptions options);
  HardtanhOptions options;
};

TORCH_MODULE(Hardtanh);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct LeakyReLUOptions {
  LeakyReLUOptions(double negative_slope);
  TORCH_ARG(double, negative_slope);
};

struct LeakyReLUImpl : FunctionalImpl {
  explicit LeakyReLUImpl(LeakyReLUOptions options);
  LeakyReLUOptions options;
};

TORCH_MODULE(LeakyReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softplus ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct SoftplusOptions {
  SoftplusOptions(double beta);
  TORCH_ARG(double, beta);
  TORCH_ARG(double, threshold) = 20;
};

struct SoftplusImpl : FunctionalImpl {
  explicit SoftplusImpl(SoftplusOptions options);
  SoftplusOptions options;
};

TORCH_MODULE(Softplus);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softsign ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct SoftsignImpl : FunctionalImpl {
  SoftsignImpl();
};

TORCH_MODULE(Softsign);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct SoftmaxOptions {
  SoftmaxOptions(int64_t dim);
  TORCH_ARG(int64_t, dim);
};

struct SoftmaxImpl : FunctionalImpl {
  explicit SoftmaxImpl(SoftmaxOptions options);
  SoftmaxOptions options;
};

TORCH_MODULE(Softmax);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSoftmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct LogSoftmaxOptions {
  LogSoftmaxOptions(int64_t dim);
  TORCH_ARG(int64_t, dim);
};

struct LogSoftmaxImpl : FunctionalImpl {
  explicit LogSoftmaxImpl(LogSoftmaxOptions options);
  LogSoftmaxOptions options;
};

TORCH_MODULE(LogSoftmax);
} // namespace nn
} // namespace torch
