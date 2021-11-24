#include <ATen/native/Unfold3d.h>

namespace at {
namespace native {

void Unfold3dCopyCUDA(
    ScalarType dtype,
    const void *src,
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    void* dst) {
  AT_ERROR("Unfold3dCopy is not currently supported for CUDA");
}

void Unfold3dAccCUDA(
    ScalarType dtype,
    const void *src,
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    void* dst) {
  AT_ERROR("Unfold3dAcc is not currently supported for CUDA");
}

REGISTER_DISPATCH(unfolded3d_copy_stub, &Unfold3dCopyCUDA);
REGISTER_DISPATCH(unfolded3d_acc_stub, &Unfold3dAccCUDA);

} // namespace native
} // namespace at
