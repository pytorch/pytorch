#pragma once

#include <c10/util/Exception.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/WrapDimMinimal.h>

/* Each tensor has a dimension layout in physical memory after it is allocated.
 * For eg, a channels-last tensor is allocated in memory as NHWC, where the last
 * dimension is channel (index is 3), NHWC is the dimension layout in physical
 * memory. However, we can use a logical layout to represent this channels-last
 * tensor, not necessarily the same as NHWC, such as NCHW, we can get NCHW by
 * doing NHWC.permute(0, 3, 1, 2). This (0, 3, 1, 2) is called a layout
 * permutation.
 *
 * In other words, each dimension in the layout permutation indicates that,
 * given each logical layout dimension of current tensor, what is the original
 * dimension in physical layout when this tensor is allocated. Here, the index
 * of the layout permutation repreents the dimension in logical layout.
 * In above example, A channels-last tensor with NCHW logical layout has a
 * layout permutation (0, 3, 1, 2), since channel dimension is the last
 * dimension in pyhsical layout, channel dimension in logical layout is 1, but
 * it is 3 in physical layout, so we have layout permutation[1] == 3, same rule
 * applied to other dimensions.
 *
 * This class is designed to use a 64bit unsigned integer to hold a layout
 * permutation, 4bit for each dimension, it can support a max of 15 dimensions.
 * (1111 is reserved to indicate 'unknown'). eg. layout permutation (0, 3, 1, 2)
 * is translated to 0xffffffffffff0312. 'unknown' is 0xffffffffffffffff.
 *
 * This class is mainly used to provide additional information on memory format
 * propergation. For example, for a shape like (N, 1, 1, 1) tensor, it is not
 * possible to tell whether it is contiguous or channels-last contiguous only
 * based on strides, the layout permutation provides additional info, in this
 * example, tensor is channels-last contiguous only when layout permutation is
 * (0, 3, 1, 2).
 */

namespace c10 {

class TensorLayoutPermutation {

  friend std::ostream& operator<<(std::ostream& os, const TensorLayoutPermutation& p);
  friend bool operator==(const TensorLayoutPermutation& p1, const TensorLayoutPermutation& p2);
  friend bool operator!=(const TensorLayoutPermutation& p1, const TensorLayoutPermutation& p2);

  public:
    TensorLayoutPermutation() = default;
    TensorLayoutPermutation(uint64_t perm) : d_perm(perm) {};

  public:
    bool is_unknown() const;
    void reset();
    uint64_t get() const;
    void set_by_ndim(int64_t ndim);
    void set_by_dims(IntArrayRef dims);
    void set_by_perm(uint64_t perm);
    bool is_valid_ndim(int64_t ndim) const;
    bool has_equal_ndim(int64_t ndim) const;

    void permute_(IntArrayRef dims);
    TensorLayoutPermutation permute(IntArrayRef dims) const;

  private:
    void check_ndim_boundary(int64_t ndim) const;
    void check_ndim_equal(int64_t ndim) const;
    uint64_t calc_permute(IntArrayRef dims) const ;
    std::string to_string() const;

  private:
    uint64_t d_perm = 0xffffffffffffffff;
};

inline
bool TensorLayoutPermutation::is_unknown() const {
  return d_perm == 0xffffffffffffffff;
}

inline
uint64_t TensorLayoutPermutation::get() const {
  return d_perm;
}

inline
void TensorLayoutPermutation::reset() {
  d_perm = 0xffffffffffffffff;
}

inline
void TensorLayoutPermutation::set_by_ndim(int64_t ndim) {
  check_ndim_boundary(ndim);
  switch(ndim) {
    case 1:  d_perm = 0xfffffffffffffff0; return;
    case 2:  d_perm = 0xffffffffffffff01; return;
    case 3:  d_perm = 0xfffffffffffff012; return;
    case 4:  d_perm = 0xffffffffffff0123; return;
    case 5:  d_perm = 0xfffffffffff01234; return;
    case 6:  d_perm = 0xffffffffff012345; return;
    case 7:  d_perm = 0xfffffffff0123456; return;
    case 8:  d_perm = 0xffffffff01234567; return;
    case 9:  d_perm = 0xfffffff012345678; return;
    case 10: d_perm = 0xffffff0123456789; return;
    case 11: d_perm = 0xfffff0123456789a; return;
    case 12: d_perm = 0xffff0123456789ab; return;
    case 13: d_perm = 0xfff0123456789abc; return;
    case 14: d_perm = 0xff0123456789abcd; return;
    case 15: d_perm = 0xf0123456789abcde; return;
    default: d_perm = 0xffffffffffffffff;
  }
}

inline
void TensorLayoutPermutation::set_by_dims(IntArrayRef dims) {
  int64_t ndim = dims.size();
  check_ndim_boundary(ndim);
  uint64_t perm = 0xffffffffffffffff;
  if (ndim > 0) {
    std::vector<bool> seen(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
      int64_t dim = maybe_wrap_dim(dims[i], ndim);
      TORCH_CHECK(!seen[dim], "duplicated dim ", dim, " in dims");
      seen[dim] = true;
      perm <<= 4;
      perm |= dim;
    }
  }
  d_perm = perm;
}

inline
void TensorLayoutPermutation::set_by_perm(uint64_t perm) {
  d_perm = perm;
}

inline
uint64_t TensorLayoutPermutation::calc_permute(IntArrayRef dims) const {
  if (is_unknown()) {
    return d_perm;
  }
  int64_t ndim = dims.size();
  check_ndim_equal(ndim);

  // dims's ndim must be > 0 here since d_perm's ndim > 0 and they are equal
  int64_t max_dim = ndim - 1;
  uint64_t new_perm = 0xffffffffffffffff << (ndim << 2);
  std::vector<bool> seen(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    int64_t dim = maybe_wrap_dim(dims[i], ndim);
    TORCH_CHECK(!seen[dim], "duplicated dim ", dim, " in permute");
    seen[dim] = true;
    new_perm |= (d_perm >> ((max_dim - dim) << 2) & 0xf) << ((max_dim - i) << 2);
  }
  return new_perm;
}

inline
TensorLayoutPermutation TensorLayoutPermutation::permute(IntArrayRef dims) const {
  return TensorLayoutPermutation(calc_permute(dims));
}

inline
void TensorLayoutPermutation::permute_(IntArrayRef dims) {
  d_perm = calc_permute(dims);
}

inline
bool TensorLayoutPermutation::is_valid_ndim(int64_t ndim) const {
  return ndim >= 0 && ndim <= 15;
}

inline
bool TensorLayoutPermutation::has_equal_ndim(int64_t ndim) const {
  if (!is_valid_ndim(ndim)) {
    return false;
  }
  if (ndim > 0) {
      uint64_t lower = 0xffffffffffffffff << (ndim << 2);
      uint64_t upper = 0xffffffffffffffff << ((ndim - 1) << 2);

      return d_perm > lower && d_perm < upper;
  }
  // return false for ndim == 0 since it is the same as 'unknown'
  return false;
}

inline
void TensorLayoutPermutation::check_ndim_boundary(int64_t ndim) const {
  TORCH_CHECK(
    is_valid_ndim(ndim),
    "Layout permutation supports dimension [0,15], but got ", ndim);
}

inline
void TensorLayoutPermutation::check_ndim_equal(int64_t ndim) const {
  TORCH_CHECK(
    has_equal_ndim(ndim),
    "Layout permutation's num of dimension is not the same as ", ndim);
}

inline
std::string TensorLayoutPermutation::to_string() const {
  if (is_unknown()) {
    return "Unknown";
  }

  std::string str(")");
  for (int i = 0; i <= 15; ++i) {
    int dim = d_perm >> (i << 2) & 0xf;
    if (dim == 0xf) {
      break;
    }
    str = std::to_string(dim) + (i != 0 ? ", " : "") + str;
  }
  return "(" + str;
}

inline
std::ostream& operator<<(std::ostream& os, const TensorLayoutPermutation& p) {
  return os << p.to_string();
}

inline
bool operator==(const TensorLayoutPermutation& p1, const TensorLayoutPermutation& p2) {
  return p1.d_perm == p2.d_perm;
}

inline
bool operator!=(const TensorLayoutPermutation& p1, const TensorLayoutPermutation& p2) {
  return !(p1 == p2);
}

}
