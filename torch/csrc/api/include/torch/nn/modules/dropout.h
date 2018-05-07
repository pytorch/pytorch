#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {

class Dropout : public torch::nn::CloneableModule<Dropout> {
 public:
  Dropout(double p = 0.5) : p_(p) {
    assert(p < 1 && p >= 0);
  }
  variable_list forward(variable_list) override;

 protected:
  double p_;
};

class Dropout2d : public torch::nn::CloneableModule<Dropout2d> {
 public:
  Dropout2d(double p = 0.5) : p_(p) {
    assert(p < 1 && p >= 0);
  }
  variable_list forward(variable_list) override;

 protected:
  double p_;
};

}} // namespace torch::nn
