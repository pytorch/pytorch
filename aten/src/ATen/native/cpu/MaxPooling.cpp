#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/MaxPooling.h>

namespace at {
namespace native {

namespace {

template <typename scalar_t>
inline void max_pool1d_kernel(
    scalar_t* C10_RESTRICT op,
    const scalar_t* C10_RESTRICT ip,
    const PoolingParams1D& p) {
  for (int64_t kj = 0; kj < p.KW; ++kj) {
    int64_t oj = p.valid_kernel_start(kj);
    int64_t oe = p.valid_kernel_end(kj);
    int64_t ij = oj * p.SJ + kj * p.DJ - p.PJ;
    for (; oj < oe; ++oj, ij += p.SJ) {
      bool update_max = std::isnan(ip[ij]) || op[oj] < ip[ij];
      op[oj] = update_max ? ip[ij] : op[oj];
    }
  }
}

void max_pool1d_impl(
    Tensor& output,
    const Tensor& input,
    const PoolingParams1D& p) {
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool1d_impl", [&] {
    scalar_t* const OP = output.data_ptr<scalar_t>();
    const scalar_t* const IP = input.contiguous().data_ptr<scalar_t>();

    // Value used for padding
    constexpr scalar_t FILL = std::numeric_limits<scalar_t>::has_infinity
        ? -std::numeric_limits<scalar_t>::infinity()
        : std::numeric_limits<scalar_t>::lowest();

    at::parallel_for(0, p.NB * p.NC, 0, [&](int64_t begin, int64_t end) {
      for (int64_t it = begin; it < end; ++it) {
        scalar_t* op = OP + it * p.OW;
        const scalar_t* ip = IP + it * p.IW;
        std::fill_n(op, p.OW, FILL);
        max_pool1d_kernel(op, ip, p);
      }
    });
  });
}

} // namespace

REGISTER_DISPATCH(max_pool1d_stub, &max_pool1d_impl);

} // namespace native
} // namespace at
