#include <ATen/native/Copy.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cpu/CopyKernel.h>

namespace {

template <typename self_T, typename src_T>
void _copy__cpu(at::Tensor& self, const at::Tensor& src) {
  at::CPU_tensor_apply2<self_T, src_T>(
      self, src, [](self_T& self_val, const src_T& src_val) {
        self_val = static_cast<self_T>(
            static_cast<at::native::inter_copy_type_t<self_T>>(src_val));
      });
}

template <typename self_T>
void _copy__cpu(at::Tensor& self, const at::Tensor& src) {
  AT_CHECK(self.numel() == src.numel(), "sizes do not match");
  AT_DISPATCH_ALL_TYPES_AND_HALF(src.type(), "_copy__cpu", [&]() {
    _copy__cpu<self_T, scalar_t>(self, src);
  });
}

bool copy_transpose_valid(const at::Tensor& self, const at::Tensor& src) {
  const int MIN_SZ = 60 * 60;
  return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
      src.stride(0) == 1 && src.stride(1) == src.size(0) &&
      self.numel() >= MIN_SZ;
}

} // namespace

namespace at {
namespace native {

Tensor& _s_copy__cpu(Tensor& self, const Tensor& src, bool non_blocking) {
  if (src.type_id() != CPUTensorId()) {
    _s_copy_from(src, self, non_blocking);
    return self;
  }
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      self.type(), "_copy__cpu", [&]() { ::_copy__cpu<scalar_t>(self, src); });
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

  AT_DISPATCH_ALL_TYPES_AND_HALF(
      self.type(), "_copy_same_type_transpose_", [&]() {
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

  bool serial_path = false;
  if (self.numel() == src.numel()) {
    if (self.is_contiguous() && src.is_contiguous()) {
      copy_kernel(kCPU, self, src);
    } else if (copy_transpose_valid(self, src)) {
      _copy_same_type_transpose_(self, src);
    } else {
#ifdef _OPENMP
      if (!in_parallel_region()) {
        AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "_copy_same_type_", [&]() {
          at::CPU_tensor_parallel_apply2<scalar_t, scalar_t>(
              self, src, [](scalar_t& self_val, const scalar_t& src_val) {
                self_val = src_val;
              });
        });
      } else {
        serial_path = true;
      }
#else
      serial_path = true;
#endif
    }
  } else {
    serial_path = true;
  }

  if (serial_path) {
    AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "_copy_same_type_", [&]() {
      at::CPU_tensor_apply2<scalar_t, scalar_t>(
          self, src, [](scalar_t& self_val, const scalar_t& src_val) {
            self_val = src_val;
          });
    });
  }
}

DEFINE_DISPATCH(copy_kernel);

} // namespace native
} // namespace at
