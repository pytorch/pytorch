#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>
#include <algorithm>
#include <numeric>

namespace torch {
namespace jit {
namespace fuser {

//#define TC_DEBUG

/*
 * [Note - TensorContiguity implementation]
 *
 * contiguity_
 *   stores contiguity information for each dimension:
 *   number stored should be between 0 to N+2;
 *     0   - this axis requires broadcasting;
 *     X   - where X belongs [-N, -1] U [1, N] means current axis is immediately
 *           outside of axis `abs(X)-1`. If X < 0, The two axis are contiguous
 *           and can be collapsed;
 *     N+1 - the fastest changing dimension, If -N-1, means its contiguous in
 *           storage (stride == 1);
 *     N+2 - Unknown;
 *
 * TODO sorted_axes_ is something hard to maintain in a meaningful way during
 *      merge, maybe we'll drop it if it's not of much help for kernel
 *      generation
 * sorted_axes_
 *   This is a helper field, list of axes sorted by their stride;
 *   Default would be `[0, 1, 2, ..., N-1]`
 *   Given sorted_axes_[i] == X means: the i-th axis should be X (if X belongs
 *   [0, N-1]). If X == -1, that means it's unknown. This could happen when we
 *   merge two TensorContiguity and their order of axes are not consistent.
 *
 * The design of TensorContiguity is to handle two things:
 *   1. Contiguity check - whether or not the contiguity information has
 *      changed. To do this, we can simply compare TensorContiguity::contiguity_
 *      between two instances;
 *   2. Kernel generation
 *      By looking at contiguity_ flag, we can make correct decision like:
 *        a. collpasing dimensions;
 *        b. kernel binding;
 *
 * merging two TensorContiguity would check their contiguity_ flag and mark
 * accordinly.
 *
 * Issues with current implementation [definitely not complete]:
 *   1. stride for size-1 dimension.
 *     Because of PyTorch implementation, stride for size-1 dimension is ill
 *     defined and can't properly express intended layout.
 */

// debug print. remove this guy!
#ifdef TC_DEBUG
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& data) {
  os << "(";
  for (auto i = data.begin(); i != data.end(); i++) {
    os << (*i);
    os << " ";
  }
  return os << ")";
}
#endif

TensorContiguity::TensorContiguity(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
#ifdef TC_DEBUG
  std::cout << "==== contiguity ====" << std::endl;
  std::cout << "sizes: " << sizes << std::endl;
  std::cout << "strides: " << strides << std::endl;
#endif

  // assert consistent dimensionality;
  assert(sizes.size() == strides.size());

  // check for size 0 tensor;
  // size 0 tensor is not handled yet and we should not treat it as broadcast
  assert(std::none_of(
      sizes.begin(), sizes.end(), [](int64_t size) { return size == 0; }));

  int dim = sizes.size();
  contiguity_.resize(dim);
  sorted_axes_.resize(dim);

  // TODO:
  if (dim <= 0) {
    return;
  }

  std::iota(sorted_axes_.begin(), sorted_axes_.end(), 0);

  // sort axes per their strides
  // It's important that we use stable sort here, as higher
  std::stable_sort(
      sorted_axes_.begin(),
      sorted_axes_.end(),
      [&strides](int64_t s_a, int64_t s_b) {
        return strides[s_a] > strides[s_b];
      });

#ifdef TC_DEBUG
  std::cout << "sorted index: " << sorted_axes_ << std::endl;
#endif

  // Update contiguity flag all the way until the second to the last;
  for (int i = 0; i < dim; i++) {
    // decending strides: strides[axis_p_1] <= stride[axis_p];
    int axis_p = sorted_axes_[i];
    int stride_p = strides[axis_p];
    if (stride_p == 0) {
      contiguity_[axis_p] = 0; // mark axis_p as broadcast
    } else {
      if (i + 1 == dim) {
        contiguity_[axis_p] = dim + 1;
        if (stride_p == 1) {
          contiguity_[axis_p] *= -1; // we mark axis_p as contiguous in memory;
        }
        break;
      }
      // Check if we should skip the check for collapsing, if:
      //   1. we are at the fastest changing dimension already.
      //      (i == dim-1)
      //   or
      //   2. the next dimension is a broadcast dimension.
      //      (strides[sorted_axes_[i+1]] == 0))
      if ((i == dim - 1) || (strides[sorted_axes_[i + 1]] == 0)) {
        // axis_p is the fastest changing dimension.
        //   dim+1 is out of range for next axis, so we would know it's the last
        //   dimension.
        contiguity_[axis_p] = dim + 1;
        if (stride_p == 1) {
          // we mark axis_p as contiguous in memory by setting it to negative.
          contiguity_[axis_p] *= -1;
        }
      } else {
        int axis_p_1 = sorted_axes_[i + 1];
        // mark axis_p_1 as the neighboring axis;
        // Notice the compensation for 1-based indexing.
        contiguity_[axis_p] = axis_p_1 + 1;

        // Check if axis_p could collapse down to axis_p_1;
        // [Note] Do NOT specialize on size-1 dimension! Two issues:
        //   Although size-1 could collapse with any dimension, that's going to
        //   be a specialization to the static shape information -> hence not an
        //   intended protocol.
        //   size-1 collapsing could also be misleading, as the size-1 axis
        //   could falsely give the impression that its neighboring axes are
        //   collapsible, while they are not.
        //   i.e. size[4, 1, 4]; stride[8, 1, 1];
        //   both axis 0 and 2 could fuse with axis 1 separately. but we cannot
        //   fuse them all together.
        if (stride_p == sizes[axis_p_1] * strides[axis_p_1]) {
          // negative number to specify it's collapsable.
          contiguity_[axis_p] *= -1; // mark axis_p as broadcast
        }
      }
    }
  }

#ifdef TC_DEBUG
  std::cout << "contiguity flag: " << contiguity_ << std::endl;
  std::cout << "==== done contiguity ====" << std::endl;
#endif
}

bool TensorContiguity::isBroadcastDim(int axis) const {
  assert(axis >= 0 && axis < rank());
  return contiguity_[axis] == 0;
}

std::vector<int> TensorContiguity::getBroadcastDims() const {
  std::vector<int> ret;
  for (decltype(contiguity_.size()) i{0}; i < contiguity_.size(); i++) {
    if (contiguity_[i] == 0) {
      ret.emplace_back(static_cast<int>(i));
    }
  }
  return ret;
}

// we are checking if axis can merge to right.
// axis_right == axis + 1,
bool TensorContiguity::canCollapseToHigher(int axis) const {
  // not necessary as to check `assert(axis < rank()-1);` as
  // canCollapseLowerHigher would assert on that;
  return canCollapseLowerHigher(axis, axis + 1);
}

int TensorContiguity::rank() const {
  return contiguity_.size();
}

bool TensorContiguity::canCollapseLowerHigher(int lower_axis, int higher_axis)
    const {
  // we are checking if axis can merge to right.
  // we mark contiguity_ as -(target_axis + 1), if it's collapsible;
  assert(
      lower_axis >= 0 && lower_axis < rank() && higher_axis >= 0 &&
      higher_axis < rank());
  return contiguity_[lower_axis] == -(higher_axis + 1);
}

int TensorContiguity::getFCD() const {
  for (decltype(contiguity_.size()) i{0}; i < contiguity_.size(); i++) {
    if (contiguity_[i] == (-((int)contiguity_.size()) - 1))
      return i;
  }
  return -1;
}

bool TensorContiguity::isIdentical(const TensorContiguity& tc) const {
  for (int i = 0; i < rank(); i++) {
    if (tc.contiguity_[i] != contiguity_[i]) {
      return false;
    }
  }
  return true;
}

bool TensorContiguity::isCompatible(const TensorContiguity& tc) const {
  assert(false); // not yet implemented;
  return false;
}
bool TensorContiguity::hasContiguousFCD() const {
  for (decltype(contiguity_.size()) i{0}; i < contiguity_.size(); i++) {
    if (contiguity_[i] == (-((int)contiguity_.size()) - 1))
      return true;
  }
  return false;
}

int TensorContiguity::getAxisByStride(int order) const {
  assert(order >= 0 && order < rank());
  return sorted_axes_[order];
}

const std::vector<int>& TensorContiguity::getAxesOrderedByStride() const {
  return sorted_axes_;
}

const std::vector<int>& TensorContiguity::getContiguityTag() const {
  return contiguity_;
}

const std::vector<int>& TensorContiguity::getSortedAxesTag() const {
  return sorted_axes_;
}

void TensorContiguity::merge(const TensorContiguity& tc) {
  // TODO: different rank not supported yet; This could be done if we follow
  //       numpy broadcasting rule across multiple operands. We simply insert
  //       dummy dimensions at the left for tc with lower rank()
  // see [Note - TensorContiguity implementation]
  int dim = rank();
  assert(dim == tc.rank());

  for (int i = 0; i < dim; i++) {
    int cont_flag = tc.contiguity_[i];

    if (cont_flag != contiguity_[i]) {
      if (cont_flag == -contiguity_[i]) {
        // If sorting should remain, we preserve the information but only relax
        // the contiguity information;
        contiguity_[i] = std::abs(cont_flag);
      } else {
        // mark contiguity as unknown otherwise.
        contiguity_[i] = dim + 2;
      }
      cont_flag = contiguity_[i];
    }

    // TODO: can we update sorted_axes_ information via contiguity flag?
    if (tc.sorted_axes_[i] != sorted_axes_[i]) {
      // mark sorted_axes_ as unknown;
      sorted_axes_[i] = -1;
    }
  }

#ifdef TC_DEBUG
  std::cout << "merging" << std::endl;
  std::cout << "sorted index: " << sorted_axes_ << std::endl;
  std::cout << "contiguity flag: " << contiguity_ << std::endl;
#endif
}

} // namespace fuser
} // namespace jit
} // namespace torch
