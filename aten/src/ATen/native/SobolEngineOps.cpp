#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/SobolEngineOpsUtils.h>

#include <vector>

namespace at {
namespace native {

using namespace sobol_utils;

/// This is the core function to draw samples from a `SobolEngine` given
/// its state variables (`sobolstate` and `quasi`). `dimension` can be
/// inferred from `sobolstate`, but choosing to pass it explicitly to avoid
/// an extra operation to obtain the size of the first dimension of
/// `sobolstate`.
std::tuple<Tensor, Tensor> _sobol_engine_draw(const Tensor& quasi, int64_t n, const Tensor& sobolstate,
                                              int64_t dimension, int64_t num_generated, optional<ScalarType> dtype) {
  TORCH_CHECK(sobolstate.dtype() == at::kLong,
           "sobolstate needs to be of type ", at::kLong);
  TORCH_CHECK(quasi.dtype() == at::kLong,
           "quasi needs to be of type ", at::kLong);

  Tensor wquasi = quasi.clone(at::MemoryFormat::Contiguous);
  auto result_dtype = dtype.has_value() ? dtype.value() : at::kFloat;
  Tensor result = at::empty({n, dimension}, sobolstate.options().dtype(result_dtype));

  AT_DISPATCH_FLOATING_TYPES(result_dtype, "_sobol_engine_draw", [&]() -> void {
    // We deal with `data` and `strides` due to performance issues.
    int64_t l;
    int64_t* wquasi_data = wquasi.data_ptr<int64_t>();
    int64_t* sobolstate_data = sobolstate.data_ptr<int64_t>();
    scalar_t* result_data = result.data_ptr<scalar_t>();

    int64_t wquasi_stride = wquasi.stride(0);
    int64_t sobolstate_row_stride = sobolstate.stride(0), sobolstate_col_stride = sobolstate.stride(1);
    int64_t result_row_stride = result.stride(0), result_col_stride = result.stride(1);

    for (int64_t i = 0; i < n; i++, num_generated++) {
      l = rightmost_zero(num_generated);
      for (int64_t j = 0; j < dimension; j++) {
        wquasi_data[j * wquasi_stride] ^= sobolstate_data[j * sobolstate_row_stride + l * sobolstate_col_stride];
        result_data[i * result_row_stride + j * result_col_stride] = wquasi_data[j * wquasi_stride];
      }
    }
  });

  result.mul_(RECIPD);
  return std::tuple<Tensor, Tensor>(result, wquasi);
}

/// This is the core function to fast-forward a `SobolEngine` given
/// its state variables (`sobolstate` and `quasi`). `dimension` can be
/// inferred from `sobolstate`, but is passed as an argument for the same reasons
/// specified above.
Tensor& _sobol_engine_ff_(Tensor& quasi, int64_t n, const Tensor& sobolstate,
                        int64_t dimension, int64_t num_generated) {
  TORCH_CHECK(sobolstate.dtype() == at::kLong,
           "sobolstate needs to be of type ", at::kLong);
  TORCH_CHECK(quasi.dtype() == at::kLong,
           "quasi needs to be of type ", at::kLong);

  // We deal with `data` and `strides` due to performance issues.
  int64_t l;
  int64_t* quasi_data = quasi.data_ptr<int64_t>();
  int64_t* sobolstate_data = sobolstate.data_ptr<int64_t>();

  int64_t quasi_stride = quasi.stride(0);
  int64_t sobolstate_row_stride = sobolstate.stride(0), sobolstate_col_stride = sobolstate.stride(1);

  for (int64_t i = 0; i < n; i++, num_generated++) {
    l = rightmost_zero(num_generated);
    for (int64_t j = 0; j < dimension; j++) {
      quasi_data[j * quasi_stride] ^= sobolstate_data[j * sobolstate_row_stride + l * sobolstate_col_stride];
    }
  }
  return quasi;
}

/// This is an implicit function used for randomizing the state variables of the.
/// `SobolEngine`. Arguments are a randomized `sobolstate` state variables
/// and a list of random lower triangular matrices consisting of 0s and 1s. `dimension` is
/// passed explicitly again.
Tensor& _sobol_engine_scramble_(Tensor& sobolstate, const Tensor& ltm, int64_t dimension) {
  TORCH_CHECK(sobolstate.dtype() == at::kLong,
           "sobolstate needs to be of type ", at::kLong);

  /// Require a tensor accessor for `sobolstate`
  auto ss_a = sobolstate.accessor<int64_t, 2>();

  /// For every tensor in the list of tensors, the diagonals are made 1
  /// Require a dot product of every row with a specific vector of each of the matrices in `ltm`.
  /// Instead, we perform an element-wise product of all the matrices and sum over the last dimension.
  /// The required product of the m^{th} row in the d^{th} square matrix in `ltm` can be accessed
  /// using ltm_d_a[d][m] m and d are zero-indexed
  Tensor diag_true = ltm.clone(at::MemoryFormat::Contiguous);
  diag_true.diagonal(0, -2, -1).fill_(1);
  Tensor ltm_dots = cdot_pow2(diag_true);
  auto ltm_d_a = ltm_dots.accessor<int64_t, 2>();

  /// Main scrambling loop
  for (int64_t d = 0; d < dimension; ++d) {
    for (int64_t j = 0; j < MAXBIT; ++j) {
      int64_t vdj = ss_a[d][j], l = 1, t2 = 0;
      for (int64_t p = MAXBIT - 1; p >= 0; --p) {
        int64_t lsmdp = ltm_d_a[d][p];
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
  return sobolstate;
}

/// This is a core function to initialize the main state variable of a `SobolEngine`.
/// `dimension` is passed explicitly as well (see why above)
Tensor& _sobol_engine_initialize_state_(Tensor& sobolstate, int64_t dimension) {
  TORCH_CHECK(sobolstate.dtype() == at::kLong,
           "sobolstate needs to be of type ", at::kLong);

  /// Use a tensor accessor for `sobolstate`
  auto ss_a = sobolstate.accessor<int64_t, 2>();

  /// First row of `sobolstate` is all 1s
  for (int64_t m = 0; m < MAXBIT; ++m) {
    ss_a[0][m] = 1;
  }

  /// Remaining rows of sobolstate (row 2 through dim, indexed by [1:dim])
  for (int64_t d = 1; d < dimension; ++d) {
    int64_t p = poly[d];
    int64_t m = bit_length(p) - 1;

    // First m elements of row d comes from initsobolstate
    for (int64_t i = 0; i < m; ++i) {
      ss_a[d][i] = initsobolstate[d][i];
    }

    // Fill in remaining elements of v as in Section 2 (top of pg. 90) of:
    // P. Bratley and B. L. Fox. Algorithm 659: Implementing sobol's
    // quasirandom sequence generator. ACM Trans.
    // Math. Softw., 14(1):88-100, Mar. 1988.
    for (int64_t j = m; j < MAXBIT; ++j) {
      int64_t newv = ss_a[d][j - m];
      int64_t pow2 = 1;
      for (int64_t k = 0; k < m; ++k) {
        pow2 <<= 1;
        if ((p >> (m - 1 - k)) & 1) {
          newv = newv ^ (pow2 * ss_a[d][j - k - 1]);
        }
      }
      ss_a[d][j] = newv;
    }
  }

  /// Multiply each column of sobolstate by power of 2:
  /// sobolstate * [2^(maxbit-1), 2^(maxbit-2),..., 2, 1]
  Tensor pow2s = at::pow(2, at::native::arange((MAXBIT - 1), -1, -1, sobolstate.options()));
  sobolstate.mul_(pow2s);
  return sobolstate;
}

} // namespace native
} // namespace at
