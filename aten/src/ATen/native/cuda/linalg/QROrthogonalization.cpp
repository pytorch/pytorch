#include <ATen/ATen.h>
#include <ATen/native/cuda/linalg/QROrthogonalization.cu>

template <int BLOCK_THREADS, typename scalar_t> 
void qr_main(const at::Tensor& A, at::Tensor& Q, const uint m, const uint n, const float epsilon);