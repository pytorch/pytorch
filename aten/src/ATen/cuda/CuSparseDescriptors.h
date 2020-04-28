#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/Exceptions.h>

#include <cusparse.h>

// The code style of this file mostly follows ATen/cudnn/Descriptors.h,
// with modifications cudnn -> cusparse

namespace at {
namespace native {

namespace {

template <typename T> struct CuSpValueType {};
template <> struct CuSpValueType<__half> { const cudaDataType_t type = CUDA_R_16F; };
template <> struct CuSpValueType<float> { const cudaDataType_t type = CUDA_R_32F; };
template <> struct CuSpValueType<double> { const cudaDataType_t type = CUDA_R_64F; };

template <typename T> struct CuSpIndexType {};
template <> struct CuSpIndexType<int> { const cusparseIndexType_t type = CUSPARSE_INDEX_32I; };
template <> struct CuSpIndexType<int64_t> { const cusparseIndexType_t type = CUSPARSE_INDEX_64I; };

template <typename T, cusparseStatus_t (*dtor)(T*)>
struct CuSparseDescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDASPARSE_CHECK(dtor(x));
    }
  }
};

template <typename T, cusparseStatus_t (*dtor)(T*)>
class CuSparseDescriptor {
 public:
  T* desc() const { return desc_.get(); }
  T* desc() { return desc_.get(); }

 protected:
  std::unique_ptr<T, CuSparseDescriptorDeleter<T, dtor>> desc_;
};

} // namespace

template <typename valueType>
class TORCH_CUDA_API CuSparseDnMatDescriptor
    : public CuSparseDescriptor<cusparseDnMatDescr, &cusparseDestroyDnMat> {
 public:
  CuSparseDnMatDescriptor(int64_t row, int64_t col, int64_t ldb, valueType* value) {
    constexpr cudaDataType_t valueType_ = CuSpValueType<valueType>().type;

    cusparseDnMatDescr_t raw_desc;
    TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
        &raw_desc,          /* output */
        row, col, ldb,      /* rows, cols, leading dimension */
        value,              /* values */
        valueType_,         /* data type of values */
        CUSPARSE_ORDER_COL  /* memory layout, ONLY column-major is supported now */
    ));
    desc_.reset(raw_desc);
  }
};

template <typename valueType, typename indexType>
class TORCH_CUDA_API CuSparseSpMatCsrDescriptor
    : public CuSparseDescriptor<cusparseSpMatDescr, &cusparseDestroySpMat> {
 public:
  CuSparseSpMatCsrDescriptor(
      int64_t row, int64_t col, int64_t nnz,
      indexType* csrrowptra, indexType* csrcolinda, valueType* csrvala) {
    constexpr cudaDataType_t valueType_ = CuSpValueType<valueType>().type;
    constexpr cusparseIndexType_t indexType_ = CuSpIndexType<indexType>().type;

    cusparseSpMatDescr_t raw_desc;
    TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
        &raw_desc,                  /* output */
        row, col, nnz,              /* rows, cols, number of non zero elements */
        csrrowptra,                 /* row offsets of the sparse matrix, size = rows +1 */
        csrcolinda,                 /* column indices of the sparse matrix, size = nnz */
        csrvala,                    /* values of the sparse matrix, size = nnz */
        indexType_,                 /* data type of row offsets index */
        indexType_,                 /* data type of col indices */
        CUSPARSE_INDEX_BASE_ZERO,   /* base index of row offset and col indes */
        valueType_                  /* data type of values */
    ));
    desc_.reset(raw_desc);
  }
};

} // namespace native
} // namespace at