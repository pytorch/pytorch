#include <ATen/native/Copy.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cpu/CopyKernel.h>

namespace {

bool copy_transpose_valid(const at::Tensor& self, const at::Tensor& src) {
  const int MIN_SZ = 60 * 60;
  return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
      src.stride(0) == 1 && src.stride(1) == src.size(0) &&
      self.numel() >= MIN_SZ;
}

} // namespace

namespace at {
namespace native {

Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking) {
  Tensor b_src;
  if (self.is_sparse() && src.is_sparse()) {
    return at::copy_sparse_to_sparse_(self, src, non_blocking);
  }
  if (!self.is_sparse() && !src.is_sparse()) {
    std::tie(b_src) = expand_inplace(self, src, "copy");
    return s_copy_(self, b_src, non_blocking);
  }
  AT_ERROR("copy_() between dense and sparse Tensors is not implemented! Found self type = ",
           self.type(), " and src type = ", src.type());
}

Tensor& _s_copy__cpu(Tensor& self, const Tensor& src, bool non_blocking) {
  if (src.type_id() != CPUTensorId()) {
    _s_copy_from(src, self, non_blocking);
    return self;
  }

  if (self.scalar_type() == src.scalar_type()) {
    copy_kernel_same_type(kCPU, self, src);
  } else {
    TORCH_CHECK(self.numel() == src.numel(), "sizes do not match");
    copy_kernel_cast(kCPU, self, src);
  }
  return self;
}

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's tricker
void _copy_same_type_transpose_(Tensor& self, const Tensor& src) {
  int64_t BLOCK_SZ;
  if (self.scalar_type() == kByte) {
    BLOCK_SZ = 120;
  } else {
    BLOCK_SZ = 60;
  }
  Tensor buf = empty({BLOCK_SZ, BLOCK_SZ}, self.options());

  AT_DISPATCH_ALL_TYPES_AND(
    at::ScalarType::Half, self.scalar_type(), "_copy_same_type_transpose_", [&]() {
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

void _copy_same_type__cpu(Tensor& self, const Tensor& src) {
  if (self.is_same(src)) {
    return;
  }

  if (self.numel() == src.numel() && copy_transpose_valid(self, src)) {
    return _copy_same_type_transpose_(self, src);
  }

  copy_kernel_same_type(kCPU, self, src);
}

DEFINE_DISPATCH(copy_kernel_cast);
DEFINE_DISPATCH(copy_kernel_same_type);

} // namespace native
} // namespace at
