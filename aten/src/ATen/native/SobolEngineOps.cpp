#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/SobolEngineOpsUtils.h"

#include <vector>

namespace at {
namespace native {

/// This is the core function to draw samples from a `SobolEngine` given
/// its state variables (`sobolstate` and `quasi`). `dimension` can be
/// inferred from `sobolstate`, but choosing to pass it explicitly to avoid
/// an extra operation to obtain the size of the first dimension of
/// `sobolstate`.
std::tuple<Tensor, Tensor> _sobol_engine_draw(const Tensor& quasi, int64_t n, const Tensor& sobolstate,
                                              int64_t dimension, int64_t num_generated) {
  Tensor wquasi = quasi.clone();
  std::vector<Tensor> result;
  AT_CHECK(sobolstate.type().scalarType() == at::ScalarType::Long,
           "sobolstate needs to be of type ", at::ScalarType::Long);
  AT_CHECK(wquasi.type().scalarType() == at::ScalarType::Long,
           "quasi needs to be of type ", at::ScalarType::Long);

  for (int64_t i = 0; i < n; ++i) {
    int64_t l = rightmost_zero(num_generated);
    Tensor inter_res = wquasi.__ixor__(sobolstate.select(1, l - 1));
    result.emplace_back(inter_res.clone());
    num_generated++;
  }

  return std::make_tuple(at::native::stack(result, 0).toType(at::kFloat).mul_(RECIPD), wquasi);
}

/// This is the core function to fast-forward a `SobolEngine` given
/// its state variables (`sobolstate` and `quasi`). `dimension` can be
/// inferred from `sobolstate`, but is passed as an argument for the same reasons
/// specified above.
Tensor _sobol_engine_ff(const Tensor& quasi, int64_t n, const Tensor& sobolstate,
                        int64_t dimension, int64_t num_generated) {
  Tensor wquasi = quasi.clone();
  AT_CHECK(sobolstate.type().scalarType() == at::ScalarType::Long,
           "sobolstate needs to be of type ", at::ScalarType::Long);
  AT_CHECK(wquasi.type().scalarType() == at::ScalarType::Long,
           "quasi needs to be of type ", at::ScalarType::Long);

  for (int64_t i = 0; i < n; ++i) {
    int64_t l = rightmost_zero(num_generated);
    wquasi.__ixor__(sobolstate.select(1, l - 1));
    num_generated++;
  }
  return wquasi;
}

/// This is an implicit function used for randomizing the state variables of the.
/// `SobolEngine`. Arguments are a randomized `sobolstate` state variables
/// and a list of random lower triangular matrices consisting of 0s and 1s. `dimension` is
/// passed explicitly again.
Tensor _sobol_engine_scramble(const Tensor& sobolstate, TensorList ltm, int64_t dimension) {
  Tensor wsobolstate = sobolstate.clone();
  AT_CHECK(sobolstate.type().scalarType() == at::ScalarType::Long,
           "sobolstate needs to be of type ", at::ScalarType::Long);

  // Require a tensor accessor for `sobolstate`
  auto ss_a = wsobolstate.accessor<int64_t, 2>();

  // For every tensor in the list of tensors, the diagonals are made 1
  // Require the rows of each of the matrices in `ltm`.
  // Why is this caching these rows separately okay?
  // A simple calculation show the number of slices to be dimension * MAXBIT * MAXBIT
  // Instead, by performing `dimension` unbind operations (1 per lower triangular
  // matrix), we can save some time, since MAXBIT = 30.
  // The m^{th} row in the d^{th} square matrix in `ltm` can be obtained by ltm_rows[d*MAXBIT + m]
  // m and d are zero-indexed
  Tensor diag_true = at::native::eye(MAXBIT, wsobolstate.options()) == 1;
  std::vector<Tensor> ltm_rows;
  for (int64_t d = 0; d < dimension; ++d) {
    auto chunked_ltm = at::native::unbind(at::where(diag_true, at::ones({}, wsobolstate.options()), ltm[d]), 0);
    ltm_rows.insert(ltm_rows.end(), chunked_ltm.begin(), chunked_ltm.end());
  }

  // Main scrambling loop
  for (int64_t d = 0; d < dimension; ++d) {
    for (int64_t j = 0; j < MAXBIT; ++j) {
      int64_t vdj = ss_a[d][j], l = 1, t2 = 0;
      for (int64_t p = MAXBIT - 1; p >= 0; --p) {
        int64_t lsmdp = cdot_pow2(ltm_rows[d * MAXBIT + p]);
        int64_t t1 = 0;
        for (int64_t k = 0; k < MAXBIT; ++k) {
          t1 += (bitsubseq(lsmdp, k, 1) * bitsubseq(vdj, k, 1));
        }
        t1 = t1 % 2;
        t2 = t2 + t1 * l;
        l = l << 1;
      }
      ss_a[d][j] = t2;
    }
  }
  return wsobolstate;
}

/// This is a core function to initialize the main state variable of a `SobolEngine`.
/// `dimension` is passed explicitly as well (see why above)
Tensor _sobol_engine_initialize_state(const Tensor& sobolstate, int64_t dimension) {
  Tensor wsobolstate = sobolstate.clone();
  AT_CHECK(wsobolstate.type().scalarType() == at::ScalarType::Long,
           "sobolstate needs to be of type ", at::ScalarType::Long);

  // First row of `sobolstate` is 1
  wsobolstate.select(0, 0).fill_(1);

  // Use a tensor accessor for `sobolstate`
  auto ss_a = wsobolstate.accessor<int64_t, 2>();
  for (int64_t d = 0; d < dimension; ++d) {
    int p = poly[d];
    int m = bit_length(p) - 1;

    for (int i = 0; i < m; ++i) {
      ss_a[d][i] = initsobolstate[d][i];
    }

    for (int j = m; j < MAXBIT; ++j) {
      int newv = ss_a[d][j - m];
      int pow2 = 1;
      for (int k = 0; k < m; ++k) {
        pow2 <<= 1;
        if ((p >> (m - 1 - k)) & 1) {
          newv = newv ^ (pow2 * ss_a[d][j - k - 1]);
        }
      }
      ss_a[d][j] = newv;
    }
  }

  Tensor pow2s = at::pow(2, at::native::arange((MAXBIT - 1), -1, -1, wsobolstate.options()));
  wsobolstate = wsobolstate.mul(pow2s);
  return wsobolstate;
}

} // namespace native
} // namespace at
