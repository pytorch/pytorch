#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <c10/util/MaybeOwned.h>

namespace at { namespace native {

namespace {

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

} // namespace

c10::MaybeOwned<Tensor> prepare_batch_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, int64_t& ld_tensor, bool transpose_result, int64_t m, int64_t n) {
  IntArrayRef tensor_strides = tensor.strides();
  c10::MaybeOwned<Tensor> tensor_;
  int fast_dim = transpose_result ? 2 : 1;
  int leading_dim = transpose_result ? 1 : 2;

  if (tensor_strides[fast_dim] == 1 &&
    (tensor_strides[leading_dim] >= std::max<int64_t>(1, m))) {
    transpose_tensor = false;
    tensor_ = c10::MaybeOwned<Tensor>::borrowed(tensor);
    ld_tensor = tensor_strides[leading_dim];
  } else if ((tensor_strides[leading_dim] == 1) &&
    (tensor_strides[fast_dim] >= std::max<int64_t>(1, n))) {
    transpose_tensor = true;
    tensor_ = c10::MaybeOwned<Tensor>::borrowed(tensor);
    ld_tensor = tensor_strides[fast_dim];
  } else {
    transpose_tensor = !transpose_result;
    // gemm call requires leading dimension and stride parameters to be non-zero
    bool is_stride_non_zero = tensor.strides()[1] != 0 && tensor.strides()[2] != 0;
    if (tensor.is_contiguous() && is_stride_non_zero) {
      tensor_ = c10::MaybeOwned<Tensor>::borrowed(tensor);
    } else {
      tensor_ = c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
    }
    ld_tensor = tensor_->strides()[1];
  }

  return tensor_;
}

namespace {

Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  // Make sure to keep addmm_cuda below in sync with this code; it
  // preflights a check to try to avoid actually needing to call
  // expand().
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU("addmm", args);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self__sizes;
  c10::MaybeOwned<Tensor> self_;
  if (&result != &self) {
    self_ = expand_size(self, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    self__sizes = self_->sizes();
  } else {
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    self__sizes = self_->sizes();
    TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(self__sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
    TORCH_CHECK(self__sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
  }

  if (&result != &self) {
    at::native::resize_output(result, self__sizes);
    if (beta.toComplexDouble() != 0.0) {
      at::native::copy_(result, *self_);
    }
  }


  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  bool transpose_result;
  c10::MaybeOwned<Tensor> result_ = prepare_matrix_for_cublas(result, transpose_result);
  bool transpose_mat1;
  bool transpose_mat2;
  c10::MaybeOwned<Tensor> mat1_ = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1);
  c10::MaybeOwned<Tensor> mat2_ = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2);

  if (transpose_result) {
    transpose_mat1 = !transpose_mat1;
    transpose_mat2 = !transpose_mat2;
    mat1_sizes = mat1_->sizes();
    mat2_sizes = mat2_->sizes();
  }

  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_->stride((transpose_mat1 == transpose_result) ? 1 : 0);
  int64_t mat2_ld = mat2_->stride((transpose_mat2 == transpose_result) ? 1 : 0);
  int64_t result_ld = result_->stride(transpose_result ? 0 : 1);
  at::ScalarType scalar_type = self_->scalar_type();

  if (mat1.numel() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and infs
    // should not propagate
    if (beta.toComplexDouble() == 0.) {
      return result.zero_();
    }
    // TODO: We could squeeze some perf by calling at::cuda::mul_out here instead, to bypass the dispatcher.
    // That requires some fixing some internal build dependencies though.
    return at::mul_out(
        result,
        self,
        at::native::scalar_tensor(
            beta,
            self.scalar_type(),
            c10::nullopt /* layout */,
            at::kCPU,
            c10::nullopt /* pin_memory */));
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "addmm_cuda", [&] {
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();
    scalar_t* mat1_ptr = mat1_->data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_->data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->data_ptr<scalar_t>();
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
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

Tensor& baddbmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {batch1, "batch1", 2}, {batch2, "batch2", 3}};
  checkAllSameGPU("baddbmm", args);

  IntArrayRef batch1_sizes = batch1.sizes();
  IntArrayRef batch2_sizes = batch2.sizes();
  IntArrayRef self_sizes = self.sizes();

  TORCH_CHECK(self_sizes[0] == batch1_sizes[0], "self dim 0 must match batch1 dim 0");
  TORCH_CHECK(self_sizes[0] == batch2_sizes[0], "self dim 0 must match batch2 dim 0");
  TORCH_CHECK(self_sizes[1] == batch1_sizes[1], "self dim 1 must match batch1 dim 1");
  TORCH_CHECK(self_sizes[2] == batch2_sizes[2], "self dim 2 must match batch2 dim 2");
  TORCH_CHECK(batch1_sizes[2] == batch2_sizes[1], "batch1 dim 2 must match batch2 dim 1");

  if (!result.is_same(self)) {
    result.resize_as_(self);
    if (beta.to<c10::complex<double>>() != 0.0) {
      result.copy_(self);
    }
  }

  // handle pathological cases that blas may not like
  if (result.numel() == 0) {
    return result;
  } else if (batch1_sizes[2] == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      return result.zero_();
    } else {
      return result.mul_(beta);
    }
  }

  bool transpose_result = false;
  c10::MaybeOwned<Tensor> result_;
  IntArrayRef result_strides = result.strides();
  IntArrayRef result_sizes = result.sizes();

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) || (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    result_ = c10::MaybeOwned<Tensor>::borrowed(result);
  } else if ((result_strides[2] == 1) &&
    (result_sizes[1] == 1 || (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    transpose_result = true;
    result_ = c10::MaybeOwned<Tensor>::borrowed(result);
  } else {
    result_ = c10::MaybeOwned<Tensor>::owned(result.transpose(1, 2).clone(at::MemoryFormat::Contiguous).transpose(1, 2));
  }

  int leading_dim = transpose_result ? 1 : 2;

  int64_t m = result_sizes[transpose_result ? 2 : 1];
  int64_t n = result_sizes[leading_dim];
  int64_t k = (transpose_result ? batch2 : batch1).sizes()[leading_dim];

  int64_t lda, ldb, ldc;
  bool transpose_batch1, transpose_batch2;
  auto batch1_ = prepare_batch_matrix_for_cublas(transpose_result ? batch2 : batch1, transpose_batch1, lda, transpose_result, m, k);
  auto batch2_ = prepare_batch_matrix_for_cublas(transpose_result ? batch1 : batch2, transpose_batch2, ldb, transpose_result, k, n);

  ldc = result_->strides()[leading_dim];
  int64_t num_batches = result_->sizes()[0];

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "baddbmm_cuda", [&] {
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();
    scalar_t* batch1_ptr = batch1_->data_ptr<scalar_t>();
    scalar_t* batch2_ptr = batch2_->data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->data_ptr<scalar_t>();
    at::cuda::blas::bgemm<scalar_t>(
      transpose_batch1 ? 't' : 'n',
      transpose_batch2 ? 't' : 'n',
      m, n, k,
      alpha_val,
      batch1_ptr, lda, batch1_->strides()[0],
      batch2_ptr, ldb, batch2_->strides()[0],
      beta_val,
      result_ptr, ldc, result_->strides()[0],
      num_batches
    );
  });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

} // anonymous namespace

TORCH_IMPL_FUNC(addmm_out_cuda)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  addmm_out_cuda_impl(const_cast<Tensor&>(result), self, mat1, mat2, beta, alpha);
}

Tensor& mm_out_cuda(const Tensor& self, const Tensor& mat2, Tensor& result) {
  result.resize_({ self.size(0), mat2.size(1) });
  return addmm_out_cuda_impl(result, result, self, mat2, 0, 1);
}

Tensor mm_cuda(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({ self.size(0), mat2.size(1) }, self.options());
  return addmm_out_cuda_impl(result, result, self, mat2, 0, 1);
}

Tensor& baddbmm_out_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, Tensor &result) {
  auto self_ = &result == &self
    ? c10::MaybeOwned<Tensor>::borrowed(self)
    : expand_size(self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
  {
    at::NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, *self_, batch1, batch2, beta, alpha);
  }
  namedinference::propagate_names_if_nonempty(
       result,
       namedinference::compute_baddbmm_outnames(result, batch1, batch2, self));
  return result;
}

Tensor baddbmm_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor out = at::empty({0}, self.options());
  return baddbmm_out_cuda(self, batch1, batch2, beta, alpha, out);
}

Tensor& baddbmm__cuda(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  return baddbmm_out_cuda(self, batch1, batch2, beta, alpha, self);
}

Tensor& bmm_out_cuda(const Tensor& batch1, const Tensor& batch2, Tensor &result) {
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  at::native::resize_output(result, {batch1.sizes()[0], batch1.sizes()[1], batch2.sizes()[2]});
  Scalar beta(0.0);
  Scalar alpha(1.0);
  {
    NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, result, batch1, batch2, beta, alpha);
  }
  namedinference::propagate_names_if_nonempty(
      result,
      namedinference::compute_bmm_outnames(result, batch1, batch2));
  return result;
}

Tensor bmm_cuda(const Tensor& self, const Tensor& mat2) {
  TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
  TORCH_CHECK(mat2.dim() == 3, "batch2 must be a 3D tensor");
  Tensor result = at::empty({self.sizes()[0], self.sizes()[1], mat2.sizes()[2]}, self.options());
  return native::bmm_out_cuda(self, mat2, result);
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

return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      self.scalar_type(), "dot",
      [&] {
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

namespace {

void addr_kernel_cuda(TensorIterator &iter, const Scalar& beta, const Scalar& alpha) {
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        }
      );
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
                                         iter.dtype(), "addr_cuda", [&] {
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    scalar_t zero_val(0);
    // when beta==0, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == zero_val) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return alpha_val * vec1_val * vec2_val;
        }
      );
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return beta_val * self_val + alpha_val * vec1_val * vec2_val;
        }
      );
    }
  });
}


template <int n_threads, int n_elems_per_thread, typename func_t>
C10_LAUNCH_BOUNDS_2(n_threads, n_elems_per_thread)
__global__ void _elementwise_kernel(int total_n_elems, func_t f) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  int idx = total_work_block * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < n_elems_per_thread; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += n_threads;
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
static void _launch_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
    total_n_elems >= 0 && total_n_elems <= std::numeric_limits<int32_t>::max()
  );

  dim3 block(n_threads);
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  dim3 grid((total_n_elems + total_work_block - 1) / total_work_block);

  auto stream = at::cuda::getCurrentCUDAStream();
  _elementwise_kernel<n_threads, n_elems_per_thread, func_t>
    <<<grid, block, 0, stream>>>(total_n_elems, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void _unpack_pivots_internal_kernel(
  TensorIterator& iter,
  int64_t dim_size
) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _unpack_pivots_internal_kernel(sub_iter, dim_size);
    }
    return;
  }

  auto offset_calculator = make_offset_calculator<2>(iter);

  char* unpacked_pivots_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const char* const __restrict__ pivots_ptr = reinterpret_cast<const char*>(iter.data_ptr(1));

  auto loop = [=]C10_DEVICE(int i) {
    auto offsets = offset_calculator.get(i);

    auto* unpacked_pivots_data = reinterpret_cast<int32_t*>(
      unpacked_pivots_ptr + offsets[0]);
    const auto* const __restrict__ pivots_data = reinterpret_cast<const int32_t*>(
      pivots_ptr + offsets[1]);

    // QUESTION: can we mix 64bit offsets with 32bit Iterator indexing?
    for (int64_t i = 0; i < dim_size; ++i) {
      thrust::swap(
        unpacked_pivots_data[i],
        unpacked_pivots_data[pivots_data[i]]
      );
    }
  };

  _launch_kernel<num_threads, thread_work_size>(iter.numel(), loop);
}

void unpack_pivots_cuda_kernel(
  TensorIterator& iter,
  int64_t dim_size
) {
  _unpack_pivots_internal_kernel(iter, dim_size);
}

} // anonymous namespace

REGISTER_DISPATCH(addr_stub, &addr_kernel_cuda);
REGISTER_DISPATCH(unpack_pivots_stub, &unpack_pivots_cuda_kernel);

}}
