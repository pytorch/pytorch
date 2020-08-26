#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Pooling.h>

namespace at {
namespace native {
namespace {

template <typename scalar_t>
void max_pool2d_kernel(
    scalar_t* op,
    const scalar_t* const ip,
    const int64_t begin,
    const int64_t end,
    const PoolingParams& p) {
  /*
   * For each row of the output tensor, first compute a row-wise max of the
   * input rows accessed by the current kernel window. Then, compute the max for
   * each cell of the current output row using the row-reduced buffer.
   *
   * This algorithm makes better use of the cache, reduces duplicate comparisons
   * in the case of overlapping kernel windows and facilitates vectorization.
   * The downsides are that it uses an extra buffer and will compute row-wise
   * max of every column even if it will be skipped over when striding.
   */
  using Vec = vec256::Vec256<scalar_t>;

  // Value to fill padded region with
  constexpr scalar_t FILL = std::numeric_limits<scalar_t>::has_infinity
      ? -std::numeric_limits<scalar_t>::infinity()
      : std::numeric_limits<scalar_t>::lowest();

  // Buffer to use for reductions
  __at_align32__ scalar_t buffer[p.IW]; // NOLINT

  ////////////////////////////////////////////////////////////////////////
  // ROW-WISE REDUCTION (2D)
  ////////////////////////////////////////////////////////////////////////
  for (int64_t oi = begin; oi < end; ++oi) {
    // Compute valid kernel row limits (skip padding)
    int64_t ii = oi * p.SI - p.PI;
    const int64_t ei = std::min<int64_t>(ii + p.KH * p.DI, p.IH);
    ii += (ii < 0) ? at::divup(-ii, p.DI) * p.DI : 0;

    // Variables for reduction
    const int64_t len = p.IW;
    const scalar_t* i_ptr = ip + ii * len;
    scalar_t* o_ptr = buffer;
    const int64_t stride = p.DI * len;
    const int64_t remainder = len % Vec::size();

    // Compute row-wise max for current output row
    std::fill_n(o_ptr, len, FILL);
    for (; ii < ei; ii += p.DI, i_ptr += stride) {
      for (int64_t i = 0; i <= len - Vec::size(); i += Vec::size()) {
        const Vec vals = Vec::loadu(i_ptr + i);
        const Vec max_vals = Vec::loadu(o_ptr + i);
        vec256::maximum(max_vals, vals).store(o_ptr + i);
      }
      if (remainder) {
        const int64_t offset = len - remainder;
        const Vec vals = Vec::loadu(i_ptr + offset, remainder);
        const Vec max_vals = Vec::loadu(o_ptr + offset, remainder);
        vec256::maximum(max_vals, vals).store(o_ptr + offset, remainder);
      }
    }

    ////////////////////////////////////////////////////////////////////////
    // COLUMN-WISE REDUCTION (1D - BASE CASE)
    ////////////////////////////////////////////////////////////////////////
    for (int64_t oj = 0; oj < p.OW; ++oj, ++op) {
      // Compute valid kernel column limits (skip padding)
      int64_t ij = oj * p.SJ - p.PJ;
      const int64_t ej = std::min<int64_t>(ij + p.KW * p.DJ, p.IW);
      ij += (ij < 0) ? at::divup(-ij, p.DJ) * p.DJ : 0;

      // Compute column-wise max for current output column
      *op = FILL;
      for (; ij < ej; ij += p.DJ) {
        const scalar_t val = buffer[ij];
        *op = std::isnan(val) ? val : std::max<scalar_t>(*op, val);
      }
    }
  }
}

void max_pool2d_impl(
    Tensor& output,
    const Tensor& input,
    const PoolingParams& p) {
  const int64_t work = p.NB * p.NC * p.OH;
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_impl", [&] {
    const scalar_t* const IP = input.contiguous().data_ptr<scalar_t>();
    scalar_t* const OP = output.data_ptr<scalar_t>();
    at::parallel_for(0, work, 0, [&](int64_t begin, int64_t end) {
      // Pointers to first batch/channel assigned to this thread
      const scalar_t* ip = IP + (begin / p.OH) * p.IH * p.IW;
      scalar_t* op = OP + (begin / p.OH) * p.OH * p.OW;
      // Split work per batch/channel
      int64_t remaining = end - begin;
      begin %= p.OH;
      while (remaining > 0) {
        end = std::min<int64_t>(begin + remaining, p.OH);
        max_pool2d_kernel<scalar_t>(op + begin * p.OW, ip, begin, end, p);
        ip += p.IH * p.IW;
        op += p.OH * p.OW;
        remaining -= end - begin;
        begin = 0;
      }
    });
  });
}

} // namespace

REGISTER_DISPATCH(max_pool2d_stub, &max_pool2d_impl);

} // namespace native
} // namespace at