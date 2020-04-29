#pragma once

// guard the whole file
#if !defined(_MSC_VER) && defined(__CUDACC__) && CUDART_VERSION >= 10010 // CUDA release >= 10.1 and not windows

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/Exceptions.h>

#include <cusparse.h>
#include <library_types.h>

// LIMITATION (cusparseSpMM): 
// The generic APIs are currently (CUDA 10.1) available for all platforms except Windows. 
// Using these APIs in any other systems will result in compile-time or run-time failures. 
// Their support will be extended in the next releases. 

// The code style of this file mostly follows ATen/cudnn/Descriptors.h,
// with modifications cudnn -> cusparse

namespace at { namespace cuda { namespace sparse {

template <typename T> struct CuSpValueType {};
template <> struct CuSpValueType<__half> { const cudaDataType_t type = CUDA_R_16F; };
template <> struct CuSpValueType<float> { const cudaDataType_t type = CUDA_R_32F; };
template <> struct CuSpValueType<double> { const cudaDataType_t type = CUDA_R_64F; };

template <typename T> struct CuSpIndexType {};
template <> struct CuSpIndexType<uint16_t> { const cusparseIndexType_t type = CUSPARSE_INDEX_16U; };
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
class _CuSparseDescriptor {
 public:
  T* desc() const { return desc_.get(); }
  T* desc() { return desc_.get(); }

 protected:
  std::unique_ptr<T, CuSparseDescriptorDeleter<T, dtor>> desc_;
};

template <typename valueType>
class TORCH_CUDA_API CuSparseDnMatDescriptor
    : public _CuSparseDescriptor<cusparseDnMatDescr, &cusparseDestroyDnMat> {
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

class TORCH_CUDA_API CuSparseSpMatDescriptor
    : public _CuSparseDescriptor<cusparseSpMatDescr, &cusparseDestroySpMat> {};

template <typename valueType, typename indexType>
class TORCH_CUDA_API CuSparseSpMatCsrDescriptor : public CuSparseSpMatDescriptor {
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

template <typename valueType, typename indexType>
class TORCH_CUDA_API CuSparseSpMatCooDescriptor : public CuSparseSpMatDescriptor {
 public:
  CuSparseSpMatCooDescriptor(
      int64_t row, int64_t col, int64_t nnz,
      indexType* cooRooInd, indexType* cooColInd, valueType* cooValues) {
    constexpr cudaDataType_t valueType_ = CuSpValueType<valueType>().type;
    constexpr cusparseIndexType_t indexType_ = CuSpIndexType<indexType>().type;

    cusparseSpMatDescr_t raw_desc;
    TORCH_CUDASPARSE_CHECK(cusparseCreateCoo(
        &raw_desc,                  /* output */
        row, col, nnz,              /* rows, cols, number of non zero elements */
        cooRooInd,                  /* row offsets of the sparse matrix, size = rows +1 */
        cooColInd,                  /* column indices of the sparse matrix, size = nnz */
        cooValues,                  /* values of the sparse matrix, size = nnz */
        indexType_,                 /* data type of cooRowInd and cooColInd */
        CUSPARSE_INDEX_BASE_ZERO,   /* base index of row offset and col indes */
        valueType_                  /* data type of values */
    ));
    desc_.reset(raw_desc);
  }
};

} // namespace sparse
} // namespace cuda
} // namespace at

#endif