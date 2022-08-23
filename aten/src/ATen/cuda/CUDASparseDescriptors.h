#pragma once

#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDASparse.h>

#include <c10/core/ScalarType.h>

#if defined(USE_ROCM)
#include <type_traits>
#endif

namespace at {
namespace cuda {
namespace sparse {

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

#if defined(USE_ROCM)
// hipSPARSE doesn't define this
using cusparseMatDescr = std::remove_pointer<cusparseMatDescr_t>::type;
using cusparseDnMatDescr = std::remove_pointer<cusparseDnMatDescr_t>::type;
using cusparseDnVecDescr = std::remove_pointer<cusparseDnVecDescr_t>::type;
using cusparseSpMatDescr = std::remove_pointer<cusparseSpMatDescr_t>::type;
using cusparseSpMatDescr = std::remove_pointer<cusparseSpMatDescr_t>::type;
using cusparseSpGEMMDescr = std::remove_pointer<cusparseSpGEMMDescr_t>::type;
#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()
using bsrsv2Info = std::remove_pointer<bsrsv2Info_t>::type;
using bsrsm2Info = std::remove_pointer<bsrsm2Info_t>::type;
#endif
#endif

class TORCH_CUDA_CPP_API CuSparseMatDescriptor
    : public CuSparseDescriptor<cusparseMatDescr, &cusparseDestroyMatDescr> {
 public:
  CuSparseMatDescriptor() {
    cusparseMatDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }

  CuSparseMatDescriptor(bool upper, bool unit) {
    cusparseFillMode_t fill_mode =
        upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_type =
        unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseMatDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&raw_descriptor));
    TORCH_CUDASPARSE_CHECK(cusparseSetMatFillMode(raw_descriptor, fill_mode));
    TORCH_CUDASPARSE_CHECK(cusparseSetMatDiagType(raw_descriptor, diag_type));
    descriptor_.reset(raw_descriptor);
  }
};

#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()

class TORCH_CUDA_CPP_API CuSparseBsrsv2Info
    : public CuSparseDescriptor<bsrsv2Info, &cusparseDestroyBsrsv2Info> {
 public:
  CuSparseBsrsv2Info() {
    bsrsv2Info_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseCreateBsrsv2Info(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

class TORCH_CUDA_CPP_API CuSparseBsrsm2Info
    : public CuSparseDescriptor<bsrsm2Info, &cusparseDestroyBsrsm2Info> {
 public:
  CuSparseBsrsm2Info() {
    bsrsm2Info_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseCreateBsrsm2Info(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};

#endif // AT_USE_HIPSPARSE_TRIANGULAR_SOLVE

#if AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API()

cusparseIndexType_t getCuSparseIndexType(const c10::ScalarType& scalar_type);

#if AT_USE_HIPSPARSE_GENERIC_52_API() || AT_USE_CUSPARSE_GENERIC_API()
class TORCH_CUDA_CPP_API CuSparseDnMatDescriptor
    : public CuSparseDescriptor<cusparseDnMatDescr, &cusparseDestroyDnMat> {
 public:
  explicit CuSparseDnMatDescriptor(const Tensor& input, int64_t batch_offset = -1);
};
#endif //AT_USE_HIPSPARSE_GENERIC_52_API() || AT_USE_CUSPARSE_GENERIC_API()

class TORCH_CUDA_CPP_API CuSparseDnVecDescriptor
    : public CuSparseDescriptor<cusparseDnVecDescr, &cusparseDestroyDnVec> {
 public:
  explicit CuSparseDnVecDescriptor(const Tensor& input);
};

class TORCH_CUDA_CPP_API CuSparseSpMatDescriptor
    : public CuSparseDescriptor<cusparseSpMatDescr, &cusparseDestroySpMat> {};

class TORCH_CUDA_CPP_API CuSparseSpMatCsrDescriptor
    : public CuSparseSpMatDescriptor {
 public:
  explicit CuSparseSpMatCsrDescriptor(const Tensor& input, int64_t batch_offset = -1);

#if defined(USE_ROCM) || (defined(CUDA_VERSION) && CUDA_VERSION >= 11000)
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

#if AT_USE_CUSPARSE_GENERIC_SPSV()
  void set_mat_fill_mode(bool upper) {
    cusparseFillMode_t fill_mode =
        upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER;
    TORCH_CUDASPARSE_CHECK(cusparseSpMatSetAttribute(
        this->descriptor(),
        CUSPARSE_SPMAT_FILL_MODE,
        &fill_mode,
        sizeof(fill_mode)));
  }

  void set_mat_diag_type(bool unit) {
    cusparseDiagType_t diag_type =
        unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
    TORCH_CUDASPARSE_CHECK(cusparseSpMatSetAttribute(
        this->descriptor(),
        CUSPARSE_SPMAT_DIAG_TYPE,
        &diag_type,
        sizeof(diag_type)));
  }
#endif
};

#if AT_USE_CUSPARSE_GENERIC_SPSV()
class TORCH_CUDA_CPP_API CuSparseSpSVDescriptor
    : public CuSparseDescriptor<cusparseSpSVDescr, &cusparseSpSV_destroyDescr> {
 public:
  CuSparseSpSVDescriptor() {
    cusparseSpSVDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseSpSV_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};
#endif

#if AT_USE_CUSPARSE_GENERIC_SPSM()
class TORCH_CUDA_CPP_API CuSparseSpSMDescriptor
    : public CuSparseDescriptor<cusparseSpSMDescr, &cusparseSpSM_destroyDescr> {
 public:
  CuSparseSpSMDescriptor() {
    cusparseSpSMDescr_t raw_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseSpSM_createDescr(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
};
#endif

#if (defined(USE_ROCM) && ROCM_VERSION >= 50200) || (defined(CUDA_VERSION) && CUDA_VERSION >= 11000)
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

#endif // AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API()

} // namespace sparse
} // namespace cuda
} // namespace at
