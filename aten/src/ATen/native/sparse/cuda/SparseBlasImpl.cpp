#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/cuda/CUDASparseDescriptors.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/sparse/cuda/SparseBlasImpl.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/MaybeOwned.h>

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace cuda {

namespace {

c10::MaybeOwned<Tensor> inline prepare_dense_matrix_for_cusparse(
    const Tensor& tensor) {
  if (tensor.is_non_overlapping_and_dense() ||
      is_blas_compatible_row_major_order(tensor) ||
      is_blas_compatible_column_major_order(tensor)) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> inline prepare_dense_vector_for_cusparse(
    const Tensor& tensor) {
  if (tensor.is_non_overlapping_and_dense()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

} // anonymous namespace

void addmm_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_CUSPARSE_GENERIC_API()
  TORCH_CHECK(
      false,
      "Calling addmm on a sparse GPU tensor requires compiling ",
      "PyTorch with CUDA 10.2+ (CUDA 11+ on Windows). ",
      "Please use PyTorch built with newer CUDA version.");
#else
  c10::MaybeOwned<Tensor> result_ = prepare_dense_matrix_for_cusparse(result);
  c10::MaybeOwned<Tensor> mat2_ = prepare_dense_matrix_for_cusparse(mat2);

  // Here subscript "c" stands for column-major, substript "r" stands for
  // row-major order Both orders are supported by cuSPARSE. For mixed input we
  // need to cast 'mat2' to order of 'result'. We compute
  // result = mat1 @ op(mat2) + result.
  // If order of 'mat2' and 'result' matches, the op is
  // identity; op(mat2) == mat2. If 'result' is column-major and 'mat2' is
  // row-major we pass 'mat2' as column-major and compute
  // result_c = mat1 @ transpose(mat2_c) + result_c; mat2_r==transpose(mat2_c)
  // if 'result' is row-major and 'mat2' is column-major we pass 'mat2'
  // as row-major and compute
  // result_r = mat1 @ transpose(mat2_r) + result_r; mat2_c==transpose(mat2_r)
  IntArrayRef result_strides = result_->strides();
  IntArrayRef mat2_strides = mat2_->strides();
  auto ndim = result_->dim();
  bool is_result_row_major = (result_strides[ndim - 1] == 1);
  bool is_mat2_row_major = (mat2_strides[ndim - 1] == 1);
  bool transpose_B = false;
  if (!is_result_row_major && is_mat2_row_major) {
    transpose_B = true;
  } else if (is_result_row_major && !is_mat2_row_major) {
    transpose_B = true;
  }

  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = transpose_B ? CUSPARSE_OPERATION_TRANSPOSE
                                        : CUSPARSE_OPERATION_NON_TRANSPOSE;

  // TODO: update this to support COO sparse layout
  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(mat1);
  auto descB = at::cuda::sparse::CuSparseDnMatDescriptor(
      transpose_B ? mat2_->transpose(-2, -1) : *mat2_);
  auto descC = at::cuda::sparse::CuSparseDnMatDescriptor(*result_);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      result.scalar_type(),
      "addmm_out_sparse_csr_impl_cuda",
      [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        cudaDataType compute_type = at::cuda::getCudaDataType<scalar_t>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();

#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
        auto algorithm = CUSPARSE_MM_ALG_DEFAULT;
#else
        // TODO: update this to support COO sparse layout
        auto algorithm = CUSPARSE_SPMM_CSR_ALG2;
#endif

        size_t buffer_size;
        TORCH_CUDASPARSE_CHECK(cusparseSpMM_bufferSize(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            algorithm,
            &buffer_size // output
            ));

        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        auto work_data = allocator.allocate(buffer_size);

        TORCH_CUDASPARSE_CHECK(cusparseSpMM(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            algorithm,
            work_data.get()));
      });

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#endif
}

/*
  Computes a sparse matrix-dense vector product defined as
  y <- alpha*op(A)*x + beta*y

  Args:
  * `mat` - Tensor storing sparse m x n matrix A.
  * `vec` - Tensor storing dense vector x of size n.
  * `result` - [in] Tensor storing dense vector y of size m.
               [out] result of the operation.
*/
void addmv_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_CUSPARSE_GENERIC_API()
  TORCH_CHECK(
      false,
      "Calling addmv on a sparse GPU tensor requires compiling ",
      "PyTorch with CUDA 10.2+ (CUDA 11+ on Windows). ",
      "Please use PyTorch built with newer CUDA version.");
#else
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;

  c10::MaybeOwned<Tensor> result_ = prepare_dense_vector_for_cusparse(result);
  c10::MaybeOwned<Tensor> vec_ = prepare_dense_vector_for_cusparse(vec);

  // TODO: update this to support COO sparse layout
  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(mat);
  auto descX = at::cuda::sparse::CuSparseDnVecDescriptor(*vec_);
  auto descY = at::cuda::sparse::CuSparseDnVecDescriptor(*result_);

  // There is no dispatch for kHalf and kBFloat16 types because cusparse
  // computes garbage in this case, latest checked version of cuda is 11.3
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(),
      "addmv_out_sparse_csr_cuda_impl",
      [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        cudaDataType compute_type = at::cuda::getCudaDataType<scalar_t>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();

        // cusparseSpMVAlg_t was updated in cuda 11.2.1
        #if CUSPARSE_VERSION >= 11400
        cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;
        #else
        cusparseSpMVAlg_t alg = CUSPARSE_MV_ALG_DEFAULT;
        #endif

        size_t buffer_size;
        TORCH_CUDASPARSE_CHECK(cusparseSpMV_bufferSize(
            handle,
            opA,
            &alpha_,
            descA.descriptor(),
            descX.descriptor(),
            &beta_,
            descY.descriptor(),
            compute_type,
            alg,
            &buffer_size // output
            ));

        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        auto work_data = allocator.allocate(buffer_size);

        TORCH_CUDASPARSE_CHECK(cusparseSpMV(
            handle,
            opA,
            &alpha_,
            descA.descriptor(),
            descX.descriptor(),
            &beta_,
            descY.descriptor(),
            compute_type,
            alg,
            work_data.get()));
      });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#endif
}

} // namespace cuda
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
