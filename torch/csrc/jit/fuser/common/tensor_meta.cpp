#include <torch/csrc/jit/fuser/common/tensor_meta.h>

namespace torch {
namespace jit {
namespace fuser {

TensorContiguity::TensorContiguity(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
}

bool TensorContiguity::isBroadcastDim(int axis) {
  return false;
}

std::vector<int> TensorContiguity::getBroadcastDims() {
  return {};
}

bool TensorContiguity::canCollapseLeft(int axis) {
  return false;
}

std::vector<int> TensorContiguity::getCollapseLeftDims() {
  return {};
}

int TensorContiguity::rank() {
  return contiguity_.size();
}

bool TensorContiguity::canCollapseLowerToHigher(
    int lower_axis,
    int higher_axis) {
  return false;
}

int TensorContiguity::getAxisByStride(int order) {
  return 0;
}

std::vector<int> TensorContiguity::getAxesOrderedByStride() {
  return {};
}

}}} // namespace torch::jit::fuser
