#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/Resize.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include "c10/core/ScalarType.h"


// Copied from aten/src/ATen/native/cuda/Blas.cpp
namespace {

// TODO: https://github.com/pytorch/pytorch/pull/59380#pullrequestreview-725310492
c10::MaybeOwned<at::Tensor> inline resolve_conj_if_indicated(const at::Tensor& tensor, bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<at::Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<at::Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<at::Tensor> inline prepare_matrix_for_cublas(const at::Tensor& tensor, bool& transpose_tensor, bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  c10::IntArrayRef tensor_strides = tensor.strides();
  c10::IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<at::Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<at::Tensor> inline prepare_matrix_for_cublas(const at::Tensor& tensor, bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, true);
  }
  c10::IntArrayRef tensor_strides = tensor.strides();
  c10::IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<at::Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

struct cublasCommonArgs {
  cublasCommonArgs(const at::Tensor& mat1, const at::Tensor& mat2, at::Tensor& c) {
    bool transpose_result, transpose_mat1, transpose_mat2;
    result = prepare_matrix_for_cublas(c, transpose_result);
    mata = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
    matb = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);
    auto mat1_sizes = mat1.sizes();
    auto mat2_sizes = mat2.sizes();
    if (transpose_result) {
      transpose_mat1 = !transpose_mat1;
      transpose_mat2 = !transpose_mat2;
      mat1_sizes = mata->sizes();
      mat2_sizes = matb->sizes();
    }

    m = mat1_sizes[transpose_result ? 1 : 0];
    k = mat1_sizes[transpose_result ? 0 : 1];
    n = mat2_sizes[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_mat1 == transpose_result) ? 1 : 0);
    ldb = matb->stride((transpose_mat2 == transpose_result) ? 1 : 0);
    result_ld = result->stride(transpose_result ? 0 : 1);
    transa = transpose_mat1 ?  mata->is_conj() ? 'c' : 't' : 'n';
    transb = transpose_mat2 ?  matb->is_conj() ? 'c' : 't' : 'n';
  }
  char transa, transb;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<at::Tensor> mata, matb, result;
};
} // namespace


static bool _scaled_mm_allowed_device() {
    auto dprops = at::cuda::getCurrentDeviceProperties();
#ifdef USE_ROCM
    std::string device_arch = dprops->gcnArchName;
    static const std::vector<std::string> archs = {"gfx940", "gfx941", "gfx942"};
    for (std::string arch : archs) {
        size_t substring = device_arch.find(arch);
        if (substring != std::string::npos) {
            return true;
        }
    }
    return false;
#else
    return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
#endif
}

at::Tensor
_fp8_mm_cuda(const at::Tensor& mat1, const at::Tensor& mat2,
          const c10::optional<at::Tensor>& scale_a,
          const c10::optional<at::Tensor>& scale_b,
          const c10::optional<at::Tensor>& bias,
          c10::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat1.scalar_type());
  at::Tensor out = at::empty({0}, mat1.options().dtype(out_dtype_));
  at::Tensor amax = at::empty({0}, mat1.options().dtype(c10::ScalarType::Float));

  // Check sizes
  bool allowed_device = _scaled_mm_allowed_device();
  TORCH_CHECK(allowed_device, "torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+");
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");
  TORCH_CHECK(!scale_a || (scale_a->numel() == 1 && scale_a->scalar_type() == at::kFloat),
       "scale_a must be float scalar");
  TORCH_CHECK(!scale_b || (scale_b->numel() == 1 && scale_b->scalar_type() == at::kFloat),
       "scale_b must be a float scalar");
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1],
       " but got ", bias->numel());
  TORCH_CHECK(
      mat1.sizes()[1] % 16 == 0,
      "Expected trailing dimension of mat1 to be divisible by 16 ",
      "but got mat1 shape: (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      ".");
  TORCH_CHECK(mat2.sizes()[0] % 16 == 0 && mat2.sizes()[1] % 16 == 0, "mat2 shape (", mat2.sizes()[0], "x",
       mat2.sizes()[1], " must be divisible by 16");
  // Check types
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK(isFloat8Type(mat1.scalar_type()), "Expected mat1 to be Float8 matrix got ", mat1.scalar_type());
  TORCH_CHECK(isFloat8Type(mat2.scalar_type()), "Expected mat2 to be Float8 matrix got ", mat2.scalar_type());
  // Type restrictions imposed by CuBLASLt as of CUDA-12.1
  TORCH_CHECK(mat1.scalar_type() != c10::ScalarType::Float8_e5m2 || mat2.scalar_type() != c10::ScalarType::Float8_e5m2,
        "Multiplication of two Float8_e5m2 matrices is not supported");
  if (bias) {
    TORCH_CHECK(out.scalar_type() != at::kFloat, "Bias is not supported when out_dtype is set to Float32");
    TORCH_CHECK(bias->scalar_type() == c10::ScalarType::BFloat16 || bias->scalar_type() == c10::ScalarType::Half,
         "Bias must be either Half or BFloat16, but got ", bias->scalar_type());
    TORCH_CHECK((out.scalar_type() != at::kFloat && out.scalar_type() != c10::ScalarType::BFloat16) ||
          bias->scalar_type() == c10::ScalarType::BFloat16,
          "Bias must be BFloat16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
    TORCH_CHECK(out.scalar_type() != c10::ScalarType::Half || bias->scalar_type() == c10::ScalarType::Half,
          "Bias must be Float16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
  }
  {
    auto bias_ = bias.value_or(at::Tensor());
    auto scale_a_ = scale_a.value_or(at::Tensor());
    auto scale_b_ = scale_b.value_or(at::Tensor());
    at::TensorArg targs[]{{out, "out", 0}, {mat1, "mat1", 1}, {mat2, "mat2", 2},
                      {bias_, "bias", 3}, {scale_a_, "scale_a", 4},
                      {scale_b_, "scale_b", 5}};
    checkAllSameGPU(__func__, targs);
  }

  c10::IntArrayRef mat1_sizes = mat1.sizes();
  c10::IntArrayRef mat2_sizes = mat2.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

#if !defined(USE_ROCM) && !defined(_MSC_VER) || (defined(USE_ROCM) && ROCM_VERSION >= 60000)
  cublasCommonArgs args(mat1, mat2, out);
  TORCH_CHECK(args.transa == 't' && args.transb == 'n', "Only multiplication of row-major and column-major matrices is supported by cuBLASLt");
  at::cuda::blas::scaled_gemm(
      args.transa,
      args.transb,
      args.m,
      args.n,
      args.k,
      args.mata->data_ptr(),
      scale_a ? scale_a->data_ptr() : nullptr,
      args.lda,
      args.mata->scalar_type(),
      args.matb->data_ptr(),
      scale_b ? scale_b->data_ptr() : nullptr,
      args.ldb,
      args.matb->scalar_type(),
      bias ? bias->data_ptr(): nullptr,
      bias ? bias->scalar_type() : isFloat8Type(out_dtype_) ? at::ScalarType::Half : out_dtype_,
      args.result->data_ptr(),
      nullptr,
      args.result_ld,
      out_dtype_,
      amax.data_ptr(),
      use_fast_accum);
#else
  TORCH_CHECK(false, "_fp8_mm_cuda is not compiled for this platform.");
#endif

  return out;
}


// TORCH_LIBRARY(mm_fp8, m) {
//   m.def("_fp8_mm_cuda", _fp8_mm_cuda);
// }
