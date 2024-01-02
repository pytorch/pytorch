#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/MaxPooling.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

template <typename scalar_t>
inline void max_pool1d_kernel(
    scalar_t* C10_RESTRICT op,
    const scalar_t* C10_RESTRICT ip,
    const PoolingParams1D& p) {
  for (const auto kj : c10::irange(p.KW)) {
    int64_t oj = p.valid_output_start(kj);
    int64_t oe = p.valid_output_end(kj);
    int64_t ij = p.index(kj, oj);
    for (; oj < oe; ++oj, ij += p.SJ) {
      scalar_t val = ip[ij];
      bool update_max = std::isnan(val) || op[oj] < val;
      op[oj] = update_max ? val : op[oj];
    }
  }
}

void max_pool1d_impl(
    Tensor& output,
    const Tensor& input,
    const PoolingParams1D& p) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      input.scalar_type(),
      "max_pool1d_impl",
      [&] {
        const Tensor in = input.contiguous();
        scalar_t* const OP = output.data_ptr<scalar_t>();
        const scalar_t* const IP = in.data_ptr<scalar_t>();

        // Value used for padding
        scalar_t FILL = std::numeric_limits<scalar_t>::has_infinity
            ? -std::numeric_limits<scalar_t>::infinity()
            : std::numeric_limits<scalar_t>::lowest();

        at::parallel_for(0, p.NB * p.NC, 0, [&](int64_t begin, int64_t end) {
          for (const auto it : c10::irange(begin, end)) {
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

} // namespace at::native
