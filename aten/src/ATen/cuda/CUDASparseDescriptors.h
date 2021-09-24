#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDASparse.h>

#include <c10/core/ScalarType.h>

#if AT_USE_CUSPARSE_GENERIC_API()

namespace at {
namespace cuda {
namespace sparse {

cusparseIndexType_t getCuSparseIndexType(const c10::ScalarType& scalar_type);

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
  CuSparseDnMatDescriptor(const Tensor& input);
};

class TORCH_CUDA_CPP_API CuSparseDnVecDescriptor
    : public CuSparseDescriptor<cusparseDnVecDescr, &cusparseDestroyDnVec> {
 public:
  CuSparseDnVecDescriptor(const Tensor& input);
};

class TORCH_CUDA_CPP_API CuSparseSpMatDescriptor
    : public CuSparseDescriptor<cusparseSpMatDescr, &cusparseDestroySpMat> {};

class TORCH_CUDA_CPP_API CuSparseSpMatCsrDescriptor
    : public CuSparseSpMatDescriptor {
 public:
  CuSparseSpMatCsrDescriptor(const Tensor& input);

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  std::tuple<int64_t, int64_t, int64_t> get_size() {
    int64_t rows, cols, nnz;
    TORCH_CUDASPARSE_CHECK(cusparseSpMatGetSize(
        this->descriptor(),
        &rows,
        &cols,
        &nnz));
    return std::make_tuple(rows, cols, nnz);
  }

  void set_tensor(const Tensor& input) {
    auto crow_indices = input.crow_indices();
    auto col_indices = input.col_indices();
    auto values = input.values();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(crow_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(col_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
    TORCH_CUDASPARSE_CHECK(cusparseCsrSetPointers(
        this->descriptor(),
        crow_indices.data_ptr(),
        col_indices.data_ptr(),
        values.data_ptr()));
  }
#endif
};

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
class TORCH_CUDA_CPP_API CuSparseSpGEMMDescriptor
    : public CuSparseDescriptor<cusparseSpGEMMDescr, &cusparseSpGEMM_destroyDescr> {
 public:
  CuSparseSpGEMMDescriptor() {
    cusparseSpGEMMDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};
#endif

class TORCH_CUDA_CPP_API CuSparseMatDescriptor
    : public CuSparseDescriptor<cusparseMatDescr, &cusparseDestroyMatDescr> {
 public:
  CuSparseMatDescriptor() {
    cusparseMatDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

} // namespace sparse
} // namespace cuda
} // namespace at

#endif // AT_USE_CUSPARSE_GENERIC_API()
