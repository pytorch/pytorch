#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <torch/csrc/jit/fuser/common/utils.h>

#include <vector>
#include <cstdint>

namespace torch {
namespace jit {
namespace fuser {
/*
 Issues that we are not solving:
   1. strides for trivial dimensions (size-1 dimension);
   2. memory overlap / interleave;
 */
struct TORCH_API TensorContiguity {

  TensorContiguity(
      const std::vector<int64_t>& size, 
      const std::vector<int64_t>& stride);

  // gives broadcast information per axis;
  bool isBroadcastDim(int axis) const;

  // returns all axes that requires broadcast;
  std::vector<int> getBroadcastDims() const;

  // gives contiguity information per axis;
  // This basically calls to canCollapseLowerHigher(axis, axis+1);
  bool canCollapseToHigher(int axis) const;

  // return the rank of the tensor;
  int rank() const;


/*******************************************************************************
 * Future proof support
 *   we don't need these yet.
 * TODO: we probably won't need this until much later, but let's try solve
 * the problem that doesn't exist yet;
 ******************************************************************************/

  // [NOTE] the order of the argument matters:
  // canCollapseLowerHigher(x, y) differs from canCollapseLowerHigher(y, x)
  bool canCollapseLowerHigher(int lower_axis, int higher_axis) const;

  // FCD: Fast changing dimension, the dimension with smallest stride (>0).
  //   returns -1 if FCD doesn't exist (e.g. fully broadcast)
  int getFCD() const;
  // Check if FCD exist and has stride == 1.
  bool contiguousFCD() const;

  // This is used to support rational binding;
  int getAxisByStride(int order) const;
  std::vector<int> getAxesOrderedByStride() const;

  // TODO: we should encode this to a single integer with restricted rank.
  std::vector<int> getContiguityTag() const;
  std::vector<int> getSortedAxesTag() const;

  // TODO: merge two contiguity info;
  void merge(const TensorContiguity& tc);

protected:
  // contiguity_  : contiguity and broadcast;
  std::vector<int> contiguity_;

  // sorted_axes_ : axes ordered by strides (slow dimension to fast dimension).
  std::vector<int> sorted_axes_;
};

}}} // namespace torch::jit::fuser
