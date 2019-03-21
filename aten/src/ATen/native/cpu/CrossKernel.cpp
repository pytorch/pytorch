#include <ATen/native/Cross.h>

#include <numeric>
#include <iterator>
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vml.h>
namespace at { namespace native { namespace {

template<typename scalar_t>
static void apply_cross(Tensor& result, const Tensor& a, const Tensor& b, const int64_t dim) {
  int64_t total = a.numel() / 3;
  int64_t a_stride = a.stride(dim);
  int64_t b_stride = b.stride(dim);
  int64_t r_stride = result.stride(dim);

  scalar_t *a_ptr = a.data<scalar_t>();
  scalar_t *b_ptr = b.data<scalar_t>();
  scalar_t *r_ptr = result.data<scalar_t>();

  parallel_for(0, total, internal::GRAIN_SIZE, [&](int64_t s, int64_t e) {
    int64_t position_in_dims[a.dim()];
    int64_t index_in_curr_dim = s;
    int64_t start = 0;
    for (int64_t i = 0; i < a.dim(); i++) {
      if (i == dim) continue;
      position_in_dims[i] = index_in_curr_dim % a.size(i);
      start += (index_in_curr_dim % a.size(i)) * a.stride(i);
      index_in_curr_dim = index_in_curr_dim / a.size(i);
    }

    printf("position_in_dims_0 = %d position_in_dims_1 = %d\n", position_in_dims[0], position_in_dims[1]);
    printf("start = %d a_stride = %d b_stride = %d r_stride = %d\n", start, a_stride, b_stride, r_stride);

    while (s < e) {
      r_ptr[start+0*r_stride] = a_ptr[start+1*a_stride]*b_ptr[start+2*b_stride] - a_ptr[start+2*a_stride]*b_ptr[start+1*b_stride];
      r_ptr[start+1*r_stride] = a_ptr[start+2*a_stride]*b_ptr[start+0*b_stride] - a_ptr[start+0*a_stride]*b_ptr[start+2*b_stride];
      r_ptr[start+2*r_stride] = a_ptr[start+0*a_stride]*b_ptr[start+1*b_stride] - a_ptr[start+1*a_stride]*b_ptr[start+0*b_stride];
      s++;

      for (int i = 0; i < a.dim(); i++) {
        if (i == dim) {
          continue;
        }
        position_in_dims[i]++;
        start += a.stride(i);
        if (position_in_dims[i] == a.size(i)) {
          if (i != a.dim()-1) {
            start -= position_in_dims[i] * a.stride(i);
            position_in_dims[i] = 0;
          } else {
            break;
          }
        } else {
          break;
        }
      }
    }
  });
}

static void cross_kernel_impl(Tensor& result, const Tensor& a, const Tensor& b, const int64_t dim) {
  AT_DISPATCH_ALL_TYPES(result.scalar_type(), "cross", [&]() {
    apply_cross<scalar_t>(result, a, b, dim);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(cross_stub, &cross_kernel_impl);

}} // namespace at::native
