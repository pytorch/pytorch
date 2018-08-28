#include <utility>
#include "ATen/ATen.h"

namespace at { namespace native {

static inline bool isTransposeContiguous(Tensor& self) {
  return self.dim() == 2 &&
          self.stride(0) == 1 &&
          self.stride(1) == self.size(0);
}

/* gesv takes (self, A) and returns (sol, lu).
 * (i)  output tensors (sol, lu) may be same as input tensors (self, A)
 * (ii) for 2D matrices, .t_() represents their column-major format
 *
 * Before passing pointers to Lapack, we need to ensure that these pointers
 * represent Fortran-contiguous tensors in column-major format
 *
 * Cases:
 * 1) `out` has correct shape but elements do not form a contiguous
 * chunk of memory. Since shape is correct, we don't resize_ it. Instead, we
 * clone the input tensor into a buffer, use the buffer for Lapack and finally
 * copy the buffer to the output tensor.
 *
 * 2) out.t() is contiguous:
 *    (i)  &in == &out: use out.data() as is. Do nothing
 *    (ii) &in != &out: copy in.t() to out.t()
 * 3) out.t() is not contiguous:
 *    - resize_ should fix contiguity/size issues
 *    (i)  &in == &out: copy in.t().clone() to out (same tensor)
 *    (ii) &in != &out: copy in.t() to out
 */
static inline Tensor& prepareTensorsForLapack(
    const Tensor& in, Tensor& out, Tensor& temp) {
  int64_t x = in.size(0);
  int64_t y = (in.dim() == 1) ? 1 : in.size(1);
  bool out_tc = isTransposeContiguous(out);
  bool out_correct_shape =
    out.dim() == 2 && out.size(0) == x && out.size(1) == y;

  // view potential 1D `in` as 2D
  auto in_t = in.view({x, y}).t_();

  if (!out_tc && !out.is_contiguous() && out_correct_shape) {
    temp = in_t.clone().t_();
  } else if (out_tc && &in != &out) {
    out.t().resize_({y, x}).copy_(in_t);
  } else if (!out_tc) {
    out.resize_({y, x});
    if (&in == &out) {
      out.copy_(in_t.clone()).t_();
    } else {
      out.copy_(in_t).t_();
    }
  }
  // return ref to usable tensor for Lapack
  return temp.defined() ? temp : out;
}

static inline void checkInputs(const Tensor& self, const Tensor& A, bool batched) {
  if (batched) {
    if (A.size(-1) != A.size(-2)) {
      AT_ERROR("A must be batches of square matrices, "
          "but they are ", A.size(-1), " by ", A.size(-2), " matrices");
    } else if (A.size(-1) != self.size(-2)) {
      AT_ERROR("incompatible matrix sizes for matmul: each a "
          "matrix is ", A.size(-1), " by ", A.size(-1),
          " but each b matrix is ", self.size(-2), " by ", self.size(-1));
    }
  } else {
    if (A.dim() != 2) {
      AT_ERROR("A should have 2 dimensions, but has ", A.dim());
    } else if (self.dim() != 1 && self.dim() != 2) {
      AT_ERROR("B should have 1 or 2 dimensions, but has ", self.dim());
    } else if (A.size(0) != A.size(1)) {
      AT_ERROR("A must be a square matrix, but is ",
          A.size(0), " by ", A.size(1));
    } else if (A.size(0) != self.size(0)) {
      AT_ERROR("A,B size incompatible - A has ", A.size(0),
          " rows, B has ", self.size(0), " cols");
    }
  }
}

static inline void checkErrors(std::vector<int64_t> infos) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR("gesv: For batch ", i, ": Argument ",
          -info, " has illegal value");
    } else if (info > 0) {
      AT_ERROR("gesv: For batch ", i, ": U(", info, ",", info,
          ") is zero, singular U.");
    }
  }
}

}}  // namespace at::native
