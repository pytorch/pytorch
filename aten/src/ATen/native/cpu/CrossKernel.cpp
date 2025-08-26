#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Cross.h>

#include <numeric>
#include <iterator>
#include <algorithm>
#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>
namespace at::native {
namespace {

template<typename scalar_t>
static void apply_cross(const Tensor& result, const Tensor& a, const Tensor& b, const int64_t dim) {
  int64_t total = a.numel() / 3;
  int64_t a_stride = a.stride(dim);
  int64_t b_stride = b.stride(dim);
  int64_t r_stride = result.stride(dim);

  const scalar_t *a_ptr = a.const_data_ptr<scalar_t>();
  const scalar_t *b_ptr = b.const_data_ptr<scalar_t>();
  scalar_t *r_ptr = result.data_ptr<scalar_t>();

  parallel_for(0, total, internal::GRAIN_SIZE, [&](int64_t s, int64_t e) {
    const int64_t a_dim = a.dim();
    std::vector<int64_t> position_in_dims(a_dim);
    int64_t index_in_curr_dim = s;
    int64_t a_start = 0;
    int64_t b_start = 0;
    int64_t r_start = 0;
    for (const auto i : c10::irange(a.dim())) {
      if (i == dim) continue;
      position_in_dims[i] = index_in_curr_dim % a.size(i);
      a_start += (index_in_curr_dim % a.size(i)) * a.stride(i);
      b_start += (index_in_curr_dim % b.size(i)) * b.stride(i);
      r_start += (index_in_curr_dim % result.size(i)) * result.stride(i);
      index_in_curr_dim = index_in_curr_dim / a.size(i);
    }

    while (s < e) {
      r_ptr[r_start+0*r_stride] = a_ptr[a_start+1*a_stride]*b_ptr[b_start+2*b_stride] - a_ptr[a_start+2*a_stride]*b_ptr[b_start+1*b_stride];
      r_ptr[r_start+1*r_stride] = a_ptr[a_start+2*a_stride]*b_ptr[b_start+0*b_stride] - a_ptr[a_start+0*a_stride]*b_ptr[b_start+2*b_stride];
      r_ptr[r_start+2*r_stride] = a_ptr[a_start+0*a_stride]*b_ptr[b_start+1*b_stride] - a_ptr[a_start+1*a_stride]*b_ptr[b_start+0*b_stride];
      s++;

      for (const auto i : c10::irange(a.dim())) {
        if (i == dim) {
          continue;
        }
        position_in_dims[i]++;
        a_start += a.stride(i);
        b_start += b.stride(i);
        r_start += result.stride(i);
        if (position_in_dims[i] == a.size(i) && i != a.dim()-1) {
            a_start -= position_in_dims[i] * a.stride(i);
            b_start -= position_in_dims[i] * b.stride(i);
            r_start -= position_in_dims[i] * result.stride(i);
            position_in_dims[i] = 0;
        } else {
          break;
        }
      }
    }
  });
}

static void cross_kernel_impl(const Tensor& result, const Tensor& a, const Tensor& b, const int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, result.scalar_type(), "cross", [&]() {
    apply_cross<scalar_t>(result, a, b, dim);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(cross_stub, &cross_kernel_impl)

} // namespace at::native
