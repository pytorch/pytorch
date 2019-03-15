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
    int64_t counter[a.dim()];
    int64_t v = s;
    int64_t start = 0;
    for (int64_t i = 0; i < a.dim(); i++) {
      if (i == dim) continue;
      counter[i] = v % a.size(i);
      start += (v % a.size(i)) * a.stride(i);
      v = v / a.size(i);
    }

    while (s < e) {
      r_ptr[start] = a_ptr[start+a_stride]*b_ptr[start+2*b_stride] - a_ptr[start+2*a_stride]*b_ptr[start+b_stride];
      r_ptr[start+r_stride] = a_ptr[start+2*a_stride]*b_ptr[start] - a_ptr[start]*b_ptr[start+2*b_stride];
      r_ptr[start+2*r_stride] = a_ptr[start]*b_ptr[start+b_stride] - a_ptr[start+a_stride]*b_ptr[start];
      s++;

      for (int i = 0; i < a.dim(); i++) {
        if (i == dim) {
          continue;
        }
        counter[i]++;
        start += a.stride(i);
        if (counter[i] == a.size(i)) {
          if (i != a.dim()-1) {
            start -= counter[i] * a.stride(i);
            counter[i] = 0;
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
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, result.scalar_type(), "cross", [&]() {
    apply_cross<scalar_t>(result, a, b, dim);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(cross_stub, &cross_kernel_impl);

}} // namespace at::native
