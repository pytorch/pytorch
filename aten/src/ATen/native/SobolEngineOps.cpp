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
/// `sobolstate`. Note that changes made to `quasi` is inplace.
std::tuple<Tensor, int64_t> _sobol_engine_draw(int64_t n, const Tensor& sobolstate,
                                               Tensor& quasi, int64_t dimension, int64_t num_generated) {
  std::vector<Tensor> result;
  AT_CHECK(sobolstate.type().scalarType() == at::ScalarType::Int,
           "sobolstate needs to be of type ", at::ScalarType::Int);
  AT_CHECK(quasi.type().scalarType() == at::ScalarType::Int,
           "quasi needs to be of type ", at::ScalarType::Int);

  int64_t num_gen = num_generated;

  for (int64_t i = 0; i < n; i++) {
    int64_t l = rightmost_zero(num_gen);
    Tensor inter_res = quasi.__ixor__(sobolstate.select(1, l - 1));
    result.emplace_back(RECIPD * inter_res);
    num_gen++;
  }
  return std::make_tuple(at::native::cat(result, 0), num_gen);
}

/// This is the core function to fast-forward a `SobolEngine` given
/// its state variables (`sobolstate` and `quasi`). `dimension` can be
/// inferred from `sobolstate`, but is passed as an argument for the same reasons
/// specified above. Here too, changes made to `quasi` is inplace.
int64_t _sobol_engine_ff(int64_t n, const Tensor& sobolstate, Tensor &quasi,
                         int64_t dimension, int64_t num_generated) {
  AT_CHECK(sobolstate.type().scalarType() == at::ScalarType::Int,
           "sobolstate needs to be of type ", at::ScalarType::Int);
  AT_CHECK(quasi.type().scalarType() == at::ScalarType::Int,
           "quasi needs to be of type ", at::ScalarType::Int);

  int64_t num_gen = num_generated;

  for (int64_t i = 0; i < n; i++) {
    int64_t l = rightmost_zero(num_gen);
    quasi.__ixor__(sobolstate.select(1, l - 1));
    num_gen++;
  }
  return num_gen;
}

/// This is an implicit function used for randomizing the state variables of the.
/// `SobolEngine`. Arguments are a randomized `sobolstate` state variables
/// and a list of random lower triangular matrices consisting of 0s and 1s. `dimension` is
/// passed explicitly again. Changes made to the `sobolstate` state variable is inplace.
Tensor& _sobol_engine_scramble(Tensor& sobolstate, TensorList ltm, int64_t dimension) {
  AT_CHECK(sobolstate.type().scalarType() == at::ScalarType::Int,
           "sobolstate needs to be of type ", at::ScalarType::Int);

  // Require a tensor accessor for `sobolstate`
  auto ss_a = sobolstate.accessor<int, 2>();


  // For every tensor in the list of tensors, the diagonals are made 1
  // Require the rows of each of the matrices in `ltm`.
  // Why is this caching these rows separately okay?
  // A simple calculation show the number of slices to be dimension * MAXBIT * MAXBIT
  // Instead, by performing `dimension` chunking operations (1 per lower triangular
  // matrix), we can save some time, since MAXBIT = 30.
  Tensor eye_maxbit = at::native::eye(MAXBIT).toType(at::kByte);
  std::vector<std::vector<Tensor>> ltm_rows;
  for (int64_t d = 0; d < dimension; d++) {
    ltm_rows.emplace_back(at::native::chunk(at::where(eye_maxbit, at::ones({}, at::kInt), ltm[d]), MAXBIT));
  }

  // Main scrambling loop
  for (int64_t d = 0; d < dimension; d++) {
    for (int64_t j = 0; j < MAXBIT; j++) {
      int vdj = ss_a[d][j], l = 1, t2 = 0;
      for (int64_t p = MAXBIT; p >= 0; p--) {
        int lsmdp = cdot_pow2(ltm_rows[d][p].view(-1));
        int t1 = 0;
        for (int64_t k = 0; k < MAXBIT; k++) {
          t1 += (bitsubseq(lsmdp, k, 1) * bitsubseq(vdj, k, 1));
        }
        t1 = t1 % 2;
        t2 = t2 + t1 * l;
        l <<= 1;
      }
      ss_a[d][j] = t2;
    }
  }
  return sobolstate;
}

/// This is a core function to initialize the main state variable of a `SobolEngine`.
/// Changes made to `sobolstate` will be inplace. `dimension` is passed explicitly as well
/// (see why above)
Tensor& _sobol_engine_initialize_state(Tensor& sobolstate, int64_t dimension) {
  AT_CHECK(sobolstate.type().scalarType() == at::ScalarType::Int,
           "sobolstate needs to be of type ", at::ScalarType::Int);

  // First row of `sobolstate` is 1
  sobolstate.select(0, 0).fill_(1);

  // Use a tensor accessor for `sobolstate`
  auto ss_a = sobolstate.accessor<int, 2>();
  for (int64_t d = 0; d < dimension; d++) {
    int p = poly[d];
    int m = bit_length(p) - 1;

    for (int j = 0; j < m; j++) {
      ss_a[d][j] = initsobolstate[d][j];
    }

    for (int j = m; j < MAXBIT; j++) {
      int newv = initsobolstate[d][j - m];
      int pow2 = 1;
      for (int k = 0; k < m; k++) {
        pow2 <<= 1;
        if ((p >> (m - 1 - k)) & 1) {
          newv = newv ^ (pow2 * ss_a[d][j - k - 1]);
        }
      }
      sobolstate[d][j] = newv;
    }
  }

  Tensor pow2s = at::pow(2, at::native::arange((MAXBIT - 1), -1, -1, at::kInt));
  sobolstate.mul_(at::native::expand_as(pow2s, sobolstate));
  return sobolstate;
}

} // namespace native
} // namespace at
