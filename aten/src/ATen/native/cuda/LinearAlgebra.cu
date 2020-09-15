#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/cuda/CUDABlas.h>

namespace at { namespace native {

Tensor baddbmm_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  Tensor result = at::empty({0}, self.options());
  return baddbmm_out_cuda(result, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm__cuda(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return baddbmm_out_cuda(self, self, batch1, batch2, beta, alpha);
}

Tensor& bmm_out_cuda(Tensor &result, const Tensor& self, const Tensor& mat2) {
  result.resize_({ self.size(0), self.size(1), mat2.size(2) });
  auto dispatch_scalar_type = self.scalar_type();
  checked_dense_tensor_unwrap(result, "result", 0, "bmm_out", false, DeviceType::CUDA, dispatch_scalar_type);
  checked_dense_tensor_unwrap(self, "self", 1, "bmm_out", false, DeviceType::CUDA, dispatch_scalar_type);
  checked_dense_tensor_unwrap(mat2, "mat2", 2, "bmm_out", false, DeviceType::CUDA, dispatch_scalar_type);
  return at::baddbmm_out(result, result, self, mat2, 0, 1);
}

Tensor bmm_cuda(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({0}, self.options());
  return native::bmm_out_cuda(result, self, mat2);
}

Tensor prepare_matrix_for_cublas(Tensor& tensor, bool& transpose_tensor) {
  Tensor tensor_;
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();

  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    tensor_ = tensor;
    transpose_tensor = false;
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    tensor_ = tensor;
    transpose_tensor = true;
  } else {
    transpose_tensor = true;
    tensor_ = tensor.clone(at::MemoryFormat::Contiguous);
  }

  return tensor_;
}

namespace {

Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU("addmm", args);

  Tensor self_;
  if (&result != &self) {
    std::tie(self_) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm");
  } else {
    self_ = self;
  }

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self__sizes = self_.sizes();
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "mat1 dim 1 must match mat2 dim 0");
  TORCH_CHECK(self__sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
  TORCH_CHECK(self__sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");

  if (&result != &self) {
    at::native::resize_as_(result, self_);
    if (beta.toComplexDouble() != 0.0) {
      at::native::copy_(result, self_);
    }
  }

  TORCH_CHECK(result.dim() == 2 && self_.dim() == 2, "tensors must be 2-D");

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  bool transpose_result;
  Tensor result_ = prepare_matrix_for_cublas(result, transpose_result);
  bool transpose_mat1;
  bool transpose_mat2;
  Tensor mat1_ = transpose_result ? mat2 : mat1;
  Tensor mat2_ = transpose_result ? mat1 : mat2;
  mat1_ = prepare_matrix_for_cublas(mat1_, transpose_mat1);
  mat2_ = prepare_matrix_for_cublas(mat2_, transpose_mat2);

  if (transpose_result) {
    transpose_mat1 = !transpose_mat1;
    transpose_mat2 = !transpose_mat2;
    mat1_sizes = mat1_.sizes();
    mat2_sizes = mat2_.sizes();
  }

  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_.stride((transpose_mat1 == transpose_result) ? 1 : 0);
  int64_t mat2_ld = mat2_.stride((transpose_mat2 == transpose_result) ? 1 : 0);
  int64_t result_ld = result_.stride(transpose_result ? 0 : 1);
  at::ScalarType scalar_type = self_.scalar_type();

  if (mat1.numel() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and infs
    // should not propagate
    if (beta.toComplexDouble() == 0.) {
      return result.zero_();
    }
    return at::native::mul_out(result, self, at::native::scalar_tensor(beta, at::device(at::kCPU).dtype(self.scalar_type())));
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "addmm_cuda", [&] {
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();
    scalar_t* mat1_ptr = mat1_.data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_.data_ptr<scalar_t>();
    scalar_t* result_ptr = result_.data_ptr<scalar_t>();
    at::cuda::blas::gemm<scalar_t>(
      transpose_mat1 ? 't' : 'n',
      transpose_mat2 ? 't' : 'n',
      m, n, k,
      alpha_val,
      mat1_ptr, mat1_ld,
      mat2_ptr, mat2_ld,
      beta_val,
      result_ptr, result_ld
    );
  });
  if (result.data_ptr() != result_.data_ptr()) {
    result.copy_(result_);
  }
  return result;
}

} // anonymous namespace

Tensor& mm_out_cuda(Tensor& result, const Tensor& self, const Tensor& mat2) {
  result.resize_({ self.size(0), mat2.size(1) });
  return addmm_out_cuda_impl(result, result, self, mat2, 0, 1);
}

Tensor mm_cuda(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({ self.size(0), mat2.size(1) }, self.options());
  return addmm_out_cuda_impl(result, result, self, mat2, 0, 1);
}

Tensor& addmm_out_cuda(Tensor &out, const Tensor &self,
                        const Tensor &mat1, const Tensor &mat2,
                        Scalar beta, Scalar alpha) {
  {
    at::NoNamesGuard guard;
    Tensor& result = addmm_out_cuda_impl(out, self, mat1, mat2, beta, alpha);
  }
  at::namedinference::propagate_names_for_addmm(out, mat1, mat2, self);
  return out;
}

template <typename scalar_t>
__global__ void createBatchGemmBuffer3(const scalar_t** buffer1, const scalar_t ** buffer2, const scalar_t ** buffer3, scalar_t* data1,
                                       scalar_t * data2, scalar_t * data3, int64_t stride1, int64_t stride2, int64_t stride3, int64_t num_batches) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_batches) {
    buffer1[idx] = data1 + idx * stride1;
    buffer2[idx] = data2 + idx * stride2;
    buffer3[idx] = data3 + idx * stride3;
  }
}

template <typename scalar_t>
inline void baddbmm_out_cuda_kernel(
    Tensor& result_,
    Tensor& batch1_,
    Tensor& batch2_,
    Scalar beta,
    Scalar alpha,
    bool transpose_result,
    char transpose_batch1,
    char transpose_batch2,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t num_batches) {
#if (CUDA_VERSION < 8000) && !defined(__HIP_PLATFORM_HCC__)
  int64_t matrices_size = num_batches * sizeof(scalar_t*);
  TensorOptions options = TensorOptions().dtype(at::ScalarType::Byte).device(self.device());
  Tensor d_matrices1 = at::empty({matrices_size}, options);
  Tensor d_matrices2 = at::empty({matrices_size}, options);
  Tensor d_result_matrices = at::empty({matrices_size}, options);
  const int64_t block = 512;
  const int64_t grid = (num_batches + block - 1) / block;

  createBatchGemmBuffer3<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
    reinterpret_cast<const scalar_t**>(d_matrices1.data<scalar_t>()),
    reinterpret_cast<const scalar_t**>(d_matrices2.data<scalar_t>()),
    reinterpret_cast<const scalar_t**>(d_result_matrices.data<scalar_t>()),
    batch1_.data<scalar_t>(),
    batch2_.data<scalar_t>(),
    result_.data<scalar_t>(),
    batch1_.stride(0), batch2_.stride(0), result_.stride(0), num_batches);

  at::cuda::blas::gemmBatched<scalar_t>(
    transpose_batch1,
    transpose_batch2,
    result_.size(transpose_result ? 2 : 1),
    result_.size(transpose_result ? 1 : 2),
    batch1_.size(transpose_result ? 1 : 2),
    alpha.to<scalar_t>(),
    reinterpret_cast<const scalar_t**>(d_matrices1.data<scalar_t>()), lda,
    reinterpret_cast<const scalar_t**>(d_matrices2.data<scalar_t>()), ldb,
    beta.to<scalar_t>(),
    reinterpret_cast<scalar_t**>(d_result_matrices.data<scalar_t>()), ldc,
    num_batches);
#else
  at::cuda::blas::gemmStridedBatched<scalar_t>(
    transpose_batch1,
    transpose_batch2,
    result_.size(transpose_result ? 2 : 1),
    result_.size(transpose_result ? 1 : 2),
    batch1_.size(transpose_result ? 1 : 2),
    alpha.to<scalar_t>(),
    batch1_.data<scalar_t>(), lda, batch1_.stride(0),
    batch2_.data<scalar_t>(), ldb, batch2_.stride(0),
    beta.to<scalar_t>(),
    result_.data<scalar_t>(), ldc, result_.stride(0),
    num_batches);
#endif // CUDA_VERSION
}

template <>
void baddbmm_out_cuda_kernel<at::Half>(
    Tensor& result_,
    Tensor& batch1_,
    Tensor& batch2_,
    Scalar beta,
    Scalar alpha,
    bool transpose_result,
    char transpose_batch1,
    char transpose_batch2,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t num_batches) {
  typedef at::Half scalar_t;
#if (CUDA_VERSION < 9010) && !defined(__HIP_PLATFORM_HCC__)
  for (int64_t i = 0; i < num_batches; ++i) {
    at::cuda::blas::gemm<scalar_t>(
      transpose_batch1,
      transpose_batch2,
      result_.size(transpose_result ? 2 : 1),
      result_.size(transpose_result ? 1 : 2),
      batch1_.size(transpose_result ? 1 : 2),
      alpha.to<scalar_t>(),
      batch1_.data<scalar_t>() + i * batch1_.stride(0), lda,
      batch2_.data<scalar_t>() + i * batch2_.stride(0), ldb,
      beta.to<scalar_t>(),
      result_.data<scalar_t>() + i * result_.stride(0), ldc);
  }
#else
#if !defined(__HIP_PLATFORM_HCC__)
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 5){
#endif
  at::cuda::blas::gemmStridedBatched<scalar_t>(
    transpose_batch1,
    transpose_batch2,
    result_.size(transpose_result ? 2 : 1),
    result_.size(transpose_result ? 1 : 2),
    batch1_.size(transpose_result ? 1 : 2),
    alpha.to<scalar_t>(),
    batch1_.data<scalar_t>(), lda, batch1_.stride(0),
    batch2_.data<scalar_t>(), ldb, batch2_.stride(0),
    beta.to<scalar_t>(),
    result_.data<scalar_t>(), ldc, result_.stride(0),
    num_batches);
#if !defined(__HIP_PLATFORM_HCC__)
  } else {
    for (int64_t i = 0; i < num_batches; ++i) {
      at::cuda::blas::gemm<scalar_t>(
        transpose_batch1,
        transpose_batch2,
        result_.size(transpose_result ? 2 : 1),
        result_.size(transpose_result ? 1 : 2),
        batch1_.size(transpose_result ? 1 : 2),
        alpha.to<scalar_t>(),
        batch1_.data<scalar_t>() + i * batch1_.stride(0), lda,
        batch2_.data<scalar_t>() + i * batch2_.stride(0), ldb,
        beta.to<scalar_t>(),
        result_.data<scalar_t>() + i * result_.stride(0), ldc);
    }
  }
#endif
#endif // CUDA_VERSION
}

template <>
void baddbmm_out_cuda_kernel<at::BFloat16>(
    Tensor& result_,
    Tensor& batch1_,
    Tensor& batch2_,
    Scalar beta,
    Scalar alpha,
    bool transpose_result,
    char transpose_batch1,
    char transpose_batch2,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t num_batches) {
  typedef at::BFloat16 scalar_t;
#if defined(__HIP_PLATFORM_HCC__)
  at::cuda::blas::gemmStridedBatched<scalar_t>(
    transpose_batch1,
    transpose_batch2,
    result_.size(transpose_result ? 2 : 1),
    result_.size(transpose_result ? 1 : 2),
    batch1_.size(transpose_result ? 1 : 2),
    alpha.to<scalar_t>(),
    batch1_.data<scalar_t>(), lda, batch1_.stride(0),
    batch2_.data<scalar_t>(), ldb, batch2_.stride(0),
    beta.to<scalar_t>(),
    result_.data<scalar_t>(), ldc, result_.stride(0),
    num_batches);
#else
  TORCH_CHECK(false, "BgemmStridedBatched is not supported with at::BFloat16 type");
#endif // __HIP_PLATFORM_HCC__
}

Tensor& baddbmm_out_cuda_impl(Tensor &result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  Tensor b_self;
  std::tie(b_self) = expand_size(self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm_out");
  TORCH_CHECK(b_self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(b_self.size(0) == batch1.size(0), "equal number of batches expected");
  TORCH_CHECK(b_self.size(0) == batch2.size(0), "equal number of batches expected");
  TORCH_CHECK(b_self.size(1) == batch1.size(1), "wrong matrix size");
  TORCH_CHECK(b_self.size(2) == batch2.size(2), "wrong matrix size");
  TORCH_CHECK(batch1.size(2) == batch2.size(1), "wrong matrix size");

  if (&b_self != &result) {
    result.resize_as_(b_self);
    if (beta.to<double>() != 0.0) {
      result.copy_(b_self);
    }
  }

  bool transpose_result;
  char transpose_batch1;
  char transpose_batch2;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  Tensor result_;
  Tensor batch1_;
  Tensor batch2_;

  if (result.stride(1) == 1 &&
      (result.size(2) == 1 || result.stride(2) >= std::max<int64_t>(1, result.size(1)))) {
    transpose_result = false;
    result_ = result;
    ldc = result_.stride(2);
    batch1_ = batch1;
    batch2_ = batch2;

  } else if (result.stride(2) == 1 &&
      (result.size(1) == 1 || result.stride(1) >= std::max<int64_t>(1, result.size(2)))) {
    transpose_result = true;
    result_ = result;
    ldc = result_.stride(1);
    batch1_ = batch2;
    batch2_ = batch1;

  } else {
    transpose_result = false;
    result_ = result.transpose(1, 2).contiguous().transpose(1, 2);
    ldc = result_.stride(2);
    batch1_ = batch1;
    batch2_ = batch2;
  }

  const int64_t m = result.size(transpose_result ? 2 : 1);
  const int64_t n = result.size(transpose_result ? 1 : 2);
  const int64_t k = result.size(transpose_result ? 1 : 2);

  if (batch1_.stride(transpose_result ? 2 : 1) == 1 &&
      batch1_.stride(transpose_result ? 1 : 2) >= std::max<int64_t>(1, m)) {
    transpose_batch1 = 'n';
    lda = batch1_.stride(transpose_result ? 1 : 2);

  } else if (batch1_.stride(transpose_result ? 1 : 2) == 1 &&
      batch1_.stride(transpose_result ? 2 : 1) >= std::max<int64_t>(1, k)) {
    transpose_batch1 = 't';
    lda = batch1_.stride(transpose_result ? 2 : 1);

  } else {
    transpose_batch1 = transpose_result ? 'n' : 't';
    batch1_ = batch1_.contiguous();
    lda = batch1_.stride(1);
  }

  if (batch2_.stride(transpose_result ? 2 : 1) == 1 &&
      batch2_.stride(transpose_result ? 1 : 2) >= std::max<int64_t>(1, k)) {
    transpose_batch2 = 'n';
    ldb = batch2_.stride(transpose_result ? 1 : 2);

  } else if (batch2_.stride(transpose_result ? 1 : 2) == 1 &&
      batch2_.stride(transpose_result ? 2 : 1) >= std::max<int64_t>(1, n)) {
    transpose_batch2 = 't';
    ldb = batch2_.stride(transpose_result ? 2 : 1);

  } else {
    transpose_batch2 = transpose_result ? 'n' : 't';
    batch2_ = batch2_.contiguous();
    ldb = batch2_.stride(1);
  }

  int64_t num_batches = result_.size(0);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "baddbmm_cuda", [&] {
    baddbmm_out_cuda_kernel<scalar_t>(
      result_,
      batch1_,
      batch2_,
      beta,
      alpha,
      transpose_result,
      transpose_batch1,
      transpose_batch2,
      lda,
      ldb,
      ldc,
      num_batches);
  });
  result.resize_as_(result_);
  result.copy_(result_);
  return result;
}

Tensor& baddbmm_out_cuda(Tensor &out, const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  {
    at::NoNamesGuard guard;
    Tensor& result = baddbmm_out_cuda_impl(out, self, batch1, batch2, beta, alpha);
  }
  namedinference::propagate_names_if_nonempty(
      out,
      namedinference::compute_baddbmm_outnames(out, batch1, batch2, self));
  return out;
}

Tensor addmm_cuda(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                  Scalar beta, Scalar alpha) {
  Tensor out = at::empty({0}, self.options());
  addmm_out_cuda(out, self, mat1, mat2, beta, alpha);
  return out;
}

Tensor& addmm__cuda(Tensor& self, const Tensor& mat1, const Tensor& mat2,
                    Scalar beta, Scalar alpha) {
  addmm_out_cuda(self, self, mat1, mat2, beta, alpha);
  return self;
}

Tensor& addbmm_out_cuda(Tensor& out, const Tensor& self,
                        const Tensor& batch1, const Tensor& batch2,
                        Scalar beta, Scalar alpha) {
  TORCH_CHECK(batch1.dim() == 3 && batch2.dim() == 3,
              "Batch tensors should be 3D, got dimensions ", batch1.dim(),
              " and ", batch2.dim());

  Tensor self_;
  if (&out != &self) {
    std::tie(self_) = expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm");
  } else {
    self_ = self;
  }

  TORCH_CHECK(out.device() == self_.device() &&
              out.device() == batch1.device() &&
              out.device() == batch2.device(),
              "Expected all tensors to be on the same device. Found: ",
              out.device(), ", ", self_.device(), ", ",
              batch1.device(), " and ", batch2.device());
  TORCH_CHECK(self_.dim() == 2,
              "2D tensor expected, got ", self_.dim(), "D tensor for input");
  int64_t batchnum = batch1.size(0);
  int64_t m1d1 = batch1.size(1);
  int64_t innerdim = batch1.size(2);
  int64_t m2d2 = batch2.size(2);
  TORCH_CHECK(batchnum == batch2.size(0),
              "equal number of batches expected");
  TORCH_CHECK(m1d1 == self_.size(0),
              "first dimension of batch1  must match first dimension of input");
  TORCH_CHECK(m2d2 == self_.size(1),
              "second dimension of batch2 must match second dimension of input");
  TORCH_CHECK(innerdim == batch2.size(1),
              "second dimension of batch1 must match first dimension of batch2");

  if (&out != &self) {
    at::native::resize_as_(out, self_);
    if (beta.to<double>() != 0.0) {
      at::native::copy_(out, self_);
    }
  }

  for (int64_t i=0; i<batchnum; i++) {
    addmm_out_cuda(out, out, batch1[i], batch2[i], beta, alpha);
    beta = 1;
  }
  return out;
}

Tensor& addbmm__cuda(Tensor& self,
                     const Tensor& batch1, const Tensor& batch2,
                     Scalar beta, Scalar alpha) {
  addbmm_out_cuda(self, self, batch1, batch2, beta, alpha);
  return self;
}

Tensor addbmm_cuda(const Tensor& self,
                   const Tensor& batch1, const Tensor& batch2,
                   Scalar beta, Scalar alpha)
{
  Tensor out = at::empty({0}, self.options());
  addbmm_out_cuda(out, self, batch1, batch2, beta, alpha);
  return out;
}

namespace {

inline void dot_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "dot : expected both vectors to have same dtype, but found ",
      self.scalar_type(),
      " and ",
      other.scalar_type());
  TORCH_CHECK(
      self.numel() == other.numel(),
      "inconsistent tensor size, expected tensor [",
      self.numel(),
      "] and src [",
      other.numel(),
      "] to have the same number of elements, but got ",
      self.numel(),
      " and ",
      other.numel(),
      " elements respectively");
  TORCH_CHECK(
      self.device() == other.device(),
      "expected all tensors to be on the same device. Found: ",
      self.device(),
      ", ",
      other.device());
  TORCH_CHECK(
      (self.numel() <= INT_MAX) && (self.stride(0) <= INT_MAX) &&
          (other.stride(0) <= INT_MAX),
      "dot only supports n, incx, incy with the bound [val] <= %d",
      INT_MAX);
}

} // anonymous namespace

Tensor dot_cuda(const Tensor& self, const Tensor& other) {
  at::NoNamesGuard guard;

  dot_check(self, other);

  const int n = static_cast<int>(self.numel());
  int incx = static_cast<int>(self.stride(0));
  int incy = static_cast<int>(other.stride(0));
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, self.scalar_type(), "dot", [&] {
    Tensor result = at::empty({}, self.options());

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::PointerModeGuard pointerModeGuard(handle, CUBLAS_POINTER_MODE_DEVICE);
    at::cuda::blas::dot<scalar_t>(
        handle,
        n,
        self.data_ptr<scalar_t>(),
        incx,
        other.data_ptr<scalar_t>(),
        incy,
        result.data_ptr<scalar_t>());

    return result;
  });
}

Tensor vdot_cuda(const Tensor& self, const Tensor& other) {
  if (!self.is_complex()) {
    return dot_cuda(self, other);
  }

  at::NoNamesGuard guard;
  dot_check(self, other);

  const int n = static_cast<int>(self.numel());
  int incx = static_cast<int>(self.stride(0));
  int incy = static_cast<int>(other.stride(0));
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
    Tensor result = at::empty({}, self.options());

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::PointerModeGuard pointerModeGuard(
        handle, CUBLAS_POINTER_MODE_DEVICE);
    at::cuda::blas::vdot<scalar_t>(
        handle,
        n,
        self.data_ptr<scalar_t>(),
        incx,
        other.data_ptr<scalar_t>(),
        incy,
        result.data_ptr<scalar_t>());

    return result;
  });
}
} }
