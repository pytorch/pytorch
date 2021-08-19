#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>

#include <c10/core/ScalarType.h>

#if AT_USE_CUSPARSE_GENERIC_API()

namespace at {
namespace cuda {
namespace sparse {

namespace {

// If a specific GPU model does not provide native support for a given data
// type, cuSparse routines return CUSPARSE_STATUS_ARCH_MISMATCH error
void check_supported_cuda_type(cudaDataType cuda_type) {
  if (cuda_type == CUDA_R_16F) {
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        prop->major >= 5 && ((10 * prop->major + prop->minor) >= 53),
        "Sparse operations with CUDA tensors of Float16 type are not supported on GPUs with compute capability < 5.3 (current: ",
        prop->major,
        ".",
        prop->minor,
        ")");
  }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  if (cuda_type == CUDA_R_16BF) {
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        prop->major >= 8,
        "Sparse operations with CUDA tensors of BFloat16 type are not supported on GPUs with compute capability < 8.0 (current: ",
        prop->major,
        ".",
        prop->minor,
        ")");
  }
#endif
}

} // anonymous namespace

inline cusparseIndexType_t getCuSparseIndexType(const c10::ScalarType& scalar_type) {
  if (scalar_type == c10::ScalarType::Int) {
    return CUSPARSE_INDEX_32I;
  } else if (scalar_type == c10::ScalarType::Long) {
    return CUSPARSE_INDEX_64I;
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Cannot convert type ",
        scalar_type,
        " to cusparseIndexType.");
  }
}

template <typename T, cusparseStatus_t (*destructor)(T*)>
struct CuSparseDescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDASPARSE_CHECK(destructor(x));
    }
  }
};

template <typename T, cusparseStatus_t (*destructor)(T*)>
class CuSparseDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, CuSparseDescriptorDeleter<T, destructor>> descriptor_;
};

class TORCH_CUDA_CPP_API CuSparseDnMatDescriptor
    : public CuSparseDescriptor<cusparseDnMatDescr, &cusparseDestroyDnMat> {
 public:
  CuSparseDnMatDescriptor(const Tensor& input) {
    IntArrayRef input_strides = input.strides();
    IntArrayRef input_sizes = input.sizes();
    auto ndim = input.dim();
    auto rows = input_sizes[ndim - 2];
    auto cols = input_sizes[ndim - 1];

    bool is_column_major = at::native::is_blas_compatible_column_major_order(input);
    bool is_row_major = at::native::is_blas_compatible_row_major_order(input);
    TORCH_INTERNAL_ASSERT(
        is_column_major || is_row_major,
        "Expected either row or column major contiguous input.");

    auto leading_dimension = is_row_major ? input_strides[ndim - 2] : input_strides[ndim - 1];
    auto order = is_row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

    void* values_ptr = input.data_ptr();

    cudaDataType value_type = ScalarTypeToCudaDataType(input.scalar_type());
    check_supported_cuda_type(value_type);

    cusparseDnMatDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
        &raw_descriptor,
        rows,
        cols,
        leading_dimension,
        values_ptr,
        value_type,
        order));

    if (ndim > 2) {
      auto batch_count = at::native::cuda_int_cast(at::native::batchCount(input), "batch_count");
      auto batch_stride = input_strides[ndim - 3];
      TORCH_CUDASPARSE_CHECK(cusparseDnMatSetStridedBatch(
          raw_descriptor, batch_count, batch_stride));
    }

    descriptor_.reset(raw_descriptor);
  }
};

class TORCH_CUDA_CPP_API CuSparseDnVecDescriptor
    : public CuSparseDescriptor<cusparseDnVecDescr, &cusparseDestroyDnVec> {
 public:
  CuSparseDnVecDescriptor(const Tensor& input) {
    // cuSPARSE doesn't support batched vectors
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() == 1);

    // cuSPARSE doesn't support non-contiguous vectors
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_non_overlapping_and_dense());

    cudaDataType value_type = ScalarTypeToCudaDataType(input.scalar_type());
    check_supported_cuda_type(value_type);

    cusparseDnVecDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseCreateDnVec(
        &raw_descriptor,
        input.numel(),
        input.data_ptr(),
        value_type));
    descriptor_.reset(raw_descriptor);
  }
};

class TORCH_CUDA_CPP_API CuSparseSpMatDescriptor
    : public CuSparseDescriptor<cusparseSpMatDescr, &cusparseDestroySpMat> {};

class TORCH_CUDA_CPP_API CuSparseSpMatCsrDescriptor
    : public CuSparseSpMatDescriptor {
 public:
  CuSparseSpMatCsrDescriptor(const Tensor& input) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_sparse_csr());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);

    IntArrayRef input_sizes = input.sizes();
    auto ndim = input.dim();
    auto rows = input_sizes[ndim - 2];
    auto cols = input_sizes[ndim - 1];
    auto nnz = input._nnz();

    auto crow_indices = input.crow_indices();
    auto col_indices = input.col_indices();
    auto values = input.values();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(crow_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(col_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());

    cusparseIndexType_t index_type = getCuSparseIndexType(crow_indices.scalar_type());
    cudaDataType value_type = ScalarTypeToCudaDataType(input.scalar_type());
    check_supported_cuda_type(value_type);

    cusparseSpMatDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
        &raw_descriptor, // output descriptor
        rows,
        cols,
        nnz,
        crow_indices.data_ptr(), // row offsets of the sparse matrix, size = rows + 1
        col_indices.data_ptr(), // column indices of the sparse matrix, size = nnz
        values.data_ptr(), // values of the sparse matrix, size = nnz
        index_type, // data type of row offsets index
        index_type, // data type of col indices
        CUSPARSE_INDEX_BASE_ZERO, // base index of row offset and col indes
        value_type // data type of values
        ));

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    if (ndim > 2) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          at::native::batchCount(input) == at::native::batchCount(values));
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          at::native::batchCount(input) == at::native::batchCount(crow_indices));
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          at::native::batchCount(input) == at::native::batchCount(col_indices));
      auto batch_count = at::native::cuda_int_cast(at::native::batchCount(input), "batch_count");
      auto crow_indices_batch_stride = crow_indices.stride(-2);
      auto columns_values_batch_stride = values.stride(-2);
      TORCH_CUDASPARSE_CHECK(cusparseCsrSetStridedBatch(
          raw_descriptor,
          batch_count,
          crow_indices_batch_stride,
          columns_values_batch_stride));
    }
#endif

    descriptor_.reset(raw_descriptor);
  }
};

class TORCH_CUDA_CPP_API CuSparseSpMatCooDescriptor
    : public CuSparseSpMatDescriptor {
 public:
  CuSparseSpMatCooDescriptor(const Tensor& input) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_sparse());

    // PyTorch COO format is compatible with cuSPARSE COO only for 2D case
    // PyTorch stores indices with the shape (num_dims, nnz) and the value
    // tensor with shape (nnz,) but cuSPARSE expects separate arrays for each
    // row/col indices that are in strided batch format that is contiguous row
    // and col indices tensors with shape (batch_shape, nnz) and values tensor
    // with shape (batch_shape, nnz)
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() == 2);

    IntArrayRef input_sizes = input.sizes();
    auto ndim = input.dim();
    auto rows = input_sizes[ndim - 2];
    auto cols = input_sizes[ndim - 1];
    auto nnz = input._nnz();

    auto indices = input.indices();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(indices.is_contiguous());

    auto row_indices = indices[-2];
    auto col_indices = indices[-1];

    auto values = input.values();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());

    cusparseIndexType_t index_type = getCuSparseIndexType(indices.scalar_type());
    cudaDataType value_type = ScalarTypeToCudaDataType(values.scalar_type());
    check_supported_cuda_type(value_type);

    cusparseSpMatDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseCreateCoo(
        &raw_descriptor, // output descriptor
        rows,
        cols,
        nnz,
        row_indices.data_ptr(), // row indices of the sparse matrix, size = nnz
        col_indices.data_ptr(), // column indices of the sparse matrix, size = nnz
        values.data_ptr(), // values of the sparse matrix, size = nnz
        index_type, // data type of indices
        CUSPARSE_INDEX_BASE_ZERO, // base index of row and col indices
        value_type // data type of values
        ));

    descriptor_.reset(raw_descriptor);
  }
};

} // namespace sparse
} // namespace cuda
} // namespace at

#endif // AT_USE_CUSPARSE_GENERIC_API()
