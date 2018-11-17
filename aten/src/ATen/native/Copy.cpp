#include "Copy.h"

#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cpu/vec256/vec256.h"

namespace {

template <typename dst_T, typename src_T>
void copy_pointwise_(at::Tensor& dst, const at::Tensor& src) {
#ifdef _OPENMP
  if (at::in_parallel_region()) {
    at::CPU_tensor_parallel_apply2<dst_T, src_T>(
        dst, src, [](dst_T& dst_val, const src_T& src_val) {
          dst_val = static_cast<dst_T>(
              static_cast<at::native::inter_copy_type_t<dst_T>>(src_val));
        });
    return;
  }
#endif
  at::CPU_tensor_apply2<dst_T, src_T>(
      dst, src, [](dst_T& dst_val, const src_T& src_val) {
        dst_val = static_cast<dst_T>(
            static_cast<at::native::inter_copy_type_t<dst_T>>(src_val));
      });
}

template <typename dst_T>
void copy_pointwise_(at::Tensor& dst, const at::Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(src.type(), "copy_pointwise", [&]() {
    copy_pointwise_<dst_T, scalar_t>(dst, src);
  });
}

void copy_pointwise_(at::Tensor& dst, const at::Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(dst.type(), "copy_pointwise", [&]() {
    copy_pointwise_<scalar_t>(dst, src);
  });
}

template <typename dst_T, typename src_T>
void copy_vectorize_(at::Tensor& dst, const at::Tensor& src) {
  dst_T* dst_ptr = dst.data<dst_T>();
  src_T* src_ptr = src.data<src_T>();

  auto sample = [&](int64_t begin, int64_t end) {
    int64_t len = end - begin;
    dst_T* dst_seg = dst_ptr + begin;
    src_T* src_seg = src_ptr + begin;
    at::vec256::convert<src_T, dst_T>(src_seg, dst_seg, len);
  };

  at::parallel_for(0, dst.numel(), /* grain_size= */ 800, sample);
}

template <typename dst_T>
void copy_vectorize_(at::Tensor& dst, const at::Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(src.type(), "copy_vectorize", [&]() {
    copy_vectorize_<dst_T, scalar_t>(dst, src);
  });
}

void copy_vectorize_(at::Tensor& dst, const at::Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(dst.type(), "copy_vectorize", [&]() {
    copy_vectorize_<scalar_t>(dst, src);
  });
}

bool copy_transpose_valid(const at::Tensor& self, const at::Tensor& src) {
  const int MIN_SZ = 60 * 60;
  return self.type() == src.type() && self.is_contiguous() &&
      src.numel() != 0 && src.dim() == 2 && src.stride(0) == 1 &&
      src.stride(1) == src.size(0) && self.numel() >= MIN_SZ;
}

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's tricker
void copy_same_type_transpose_(at::Tensor& self, const at::Tensor& src) {
  const int64_t BLOCK_SZ = 60;
  at::Tensor buf = empty({BLOCK_SZ, BLOCK_SZ}, self.options());

  AT_DISPATCH_ALL_TYPES_AND_HALF(
      self.type(), "copy_same_type_transpose_", [&]() {
        scalar_t* sp = src.data<scalar_t>();
        scalar_t* rp = self.data<scalar_t>();
        scalar_t* bp = buf.data<scalar_t>();

        int64_t NR = src.size(0);
        int64_t NC = src.size(1);
        for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
          for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
            scalar_t* spo = sp + R + C * NR;
            scalar_t* rpo = rp + C + R * NC;

            int nr = std::min(NR - R, BLOCK_SZ);
            int nc = std::min(NC - C, BLOCK_SZ);

            // 1. copy columns from src to buf
            for (int c = 0; c < nc; c++) {
              memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(scalar_t));
            }

            // 2. transpose buf in place
            int rc_max = std::max(nr, nc);
            int rc_min = std::min(nr, nc);
            for (int r = 0; r < rc_max; r++) {
              int end = std::min(r, rc_min);
              for (int c = 0; c < end; c++) {
                scalar_t tmp = bp[r + BLOCK_SZ * c];
                bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
                bp[r * BLOCK_SZ + c] = tmp;
              }
            }

            // 3. copy rows from buf to dst
            for (int r = 0; r < nr; r++) {
              memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(scalar_t));
            }
          }
        }
      });
}

} // namespace

namespace at {
namespace native {

Tensor& _s_copy__cpu(Tensor& self, const Tensor& src, bool non_blocking) {
  if (src.is_cuda()) {
    _s_copy_from(src, self, non_blocking);
    return self;
  }

  if (self.is_same(src)) {
    return self;
  }

  AT_CHECK(self.numel() == src.numel(), "sizes do not match");

  if (self.is_contiguous() && src.is_contiguous()) {
    copy_vectorize_(self, src);
  } else if (copy_transpose_valid(self, src)) {
    copy_same_type_transpose_(self, src);
  } else {
    copy_pointwise_(self, src);
  }
  return self;
}

} // namespace native
} // namespace at
