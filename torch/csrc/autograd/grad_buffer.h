#pragma once

// The GradBuffer class accumulates a list of gradients for use by a
// "backward" function. It implements logic to avoid modiyfing the passed
// gradients in-place

#include <vector>
#include <utility>
#include <memory>
#include <THPP/THPP.h>

#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct GradBuffer {
  explicit GradBuffer(size_t size);
  GradBuffer(const GradBuffer& other) = delete;
  GradBuffer(GradBuffer&& other) = default;

  // Accumulates the gradient "var" at the specified index
  void addGrad(size_t idx, std::shared_ptr<Variable>&& var);

  int device() const;

  // Returns the gradients as a list of variables. Destroys this GradBuffer.
  static std::vector<std::shared_ptr<Variable>> variables(GradBuffer&& buffer);

private:
  std::vector<std::pair<std::unique_ptr<thpp::Tensor>, bool>> buffer;
};

}}  // namespace torch::autograd
