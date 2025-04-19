#pragma once

#include <ATen/core/symbol.h>

#include <c10/core/ScalarType.h>
#include <c10/util/Flags.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch::lazy {

/**
 * The goal of "dynamic" Nodes is to patch a hole in our tracing.
 * Previously, if a user called `sizes` on a Tensor, it would leak out
 * of our tracing system, as `sizes` returns a torch.Size or an int. To
 * prevent this from happening, we introduce DimensionNode, a new type
 * of Node that abstracts the operation of getting the dimensions of a
 * Tensor.
 *
 * Consider the following example:
 * ```
 * numel = x.shape()[0] * x.shape()[1]
 * ```
 *
 * Here, `x.shape()[i]` will be a SizeNode (subclass of DimensionNode),
 * and the multiplication of the two SizeNodes will be represented by
 * a SizeMul (also a subclass of DimensionNode). Through this, we can
 * prevent `numel` from being represented as a Python int and thus
 * burned into the Graph.
 */

class TORCH_API DimensionNode {
 public:
  virtual bool isSymbolic() const {
    return false;
  }
  virtual int64_t getDynamicValue() const {
    TORCH_CHECK(false, "NYI");
  }
  virtual int64_t getStaticValue() const {
    TORCH_CHECK(false, "NYI");
  }
  virtual ~DimensionNode() = default;
};

} // namespace torch::lazy
