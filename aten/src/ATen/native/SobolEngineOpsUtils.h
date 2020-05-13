/// This file contains some tensor-agnostic operations to be used in the
/// core functions of the `SobolEngine`
#include <ATen/ATen.h>

namespace at {
namespace native {
namespace sobol_utils {

/// Function to return the minimum of number of bits to represent the integer `n`
inline int64_t bit_length(const int64_t n) {
  int64_t nbits, nloc;
  for (nloc = n, nbits = 0; nloc > 0; nloc /= 2, nbits++);
  return nbits;
}

/// Function to get the position of the rightmost zero in the bit representation of an integer
/// This value is the zero-indexed position
inline int64_t rightmost_zero(const int64_t n) {
  int64_t z, i;
  for (z = n, i = 0; z % 2 == 1; z /= 2, i++);
  return i;
}

/// Function to get a subsequence of bits in the representation of an integer starting from
/// `pos` and of length `length`
inline int64_t bitsubseq(const int64_t n, const int64_t pos, const int64_t length) {
  return (n >> pos) & ((1 << length) - 1);
}

/// Function to perform the inner product between a batched square matrix and a power of 2 vector
inline at::Tensor cdot_pow2(const at::Tensor& bmat) {
  at::Tensor inter = at::arange(bmat.size(-1) - 1, -1, -1, bmat.options());
  inter = at::pow(2, inter).expand_as(bmat);
  return at::mul(inter, bmat).sum(-1);
}

/// All definitions below this point are data. These are constant, and should not be modified
/// without notice

constexpr int64_t MAXBIT = 30;
constexpr int64_t LARGEST_NUMBER = 1 << 30;
constexpr float RECIPD = 1.0 / LARGEST_NUMBER;

extern const int64_t poly[1111];
extern const int64_t initsobolstate[1111][13];

} // namespace sobol_utils
} // namespace native
} // namespace at
