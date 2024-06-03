#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SparseTensorUtils.h>
#include <cuda_runtime.h>
#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_sparse_matmul_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#endif

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <cusparse.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>


#if defined(__CUDACC__) && (CUSPARSE_VERSION >= 11000)
#define IS_CUSPARSE11_AVAILABLE() 1
#else
#define IS_CUSPARSE11_AVAILABLE() 0
#endif

#if IS_CUSPARSE11_AVAILABLE()
#include <library_types.h>
#endif

namespace at::native {

namespace {

using namespace at::sparse;

Tensor _to_csr_int(const Tensor& rowIndices, int64_t dim, int64_t nnz) {
  Tensor csr = at::empty({dim + 1}, CUDA(kInt));
  Tensor rowIndicesInt = at::empty({rowIndices.size(0)}, CUDA(kInt));
  rowIndicesInt.copy_(rowIndices);
  sparse::cuda::Xcoo2csr(
      rowIndicesInt.data_ptr<int32_t>(), nnz, dim, csr.data_ptr<int32_t>());
  return csr;
}


#pragma push
// NVCC complains that confirm_mult_size is not used,
// but it is used in specializations of CusparseMatrixMultiplyOp below
#pragma nv_diag_suppress 177   // Function was declared but never referenced
int confirm_mult_size(const std::vector<int>& mat1_size, const std::vector<int>& mat2_size) {
  TORCH_CHECK(
      mat1_size[1] == mat2_size[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1_size[0],
      "x",
      mat1_size[1],
      " and ",
      mat2_size[0],
      "x",
      mat2_size[1],
      ")");
  return mat1_size[1];
}
#pragma pop

void create_general_description_(cusparseMatDescr_t& description_) {
  TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&description_));
  TORCH_CUDASPARSE_CHECK(cusparseSetMatType(description_, CUSPARSE_MATRIX_TYPE_GENERAL));
  TORCH_CUDASPARSE_CHECK(cusparseSetMatIndexBase(description_, CUSPARSE_INDEX_BASE_ZERO));
}

// csrMatrixRef is used to have a representation of a raw CSR matrix representation
// comming from `sparse_sparse_matmul_cuda_kernel` function.
// Moreover this implements a RAII guard for a cusparse descriptor
template<class scalar_t>
struct csrMatrixRef {
  int* csr_indices_{nullptr};
  int* csr_pointers_{nullptr};
  scalar_t* csr_values_{nullptr};
  int nnz_{0};
  std::vector<int> size_{};

  #if IS_CUSPARSE11_AVAILABLE()
    cusparseSpMatDescr_t description_{0};
  #else
    cusparseMatDescr_t description_{0};
  #endif

  csrMatrixRef() {
    #if !IS_CUSPARSE11_AVAILABLE()
      create_general_description_(description_);
    #endif
  }

  csrMatrixRef(
      int* csr_indices,
      int* csr_pointers,
      scalar_t* csr_values,
      int nnz,
      const std::vector<int>& size)
      : csr_indices_{csr_indices},
        csr_pointers_{csr_pointers},
        csr_values_{csr_values},
        nnz_{nnz},
        size_{size} {
    #if IS_CUSPARSE11_AVAILABLE()
      cudaDataType cuda_data_type = at::cuda::getCudaDataType<scalar_t>();
      TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
        &description_,
        this->size(0),
        this->size(1),
        this->nnz_,
        this->csr_pointers_,
        this->csr_indices_,
        this->csr_values_,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        cuda_data_type));
    #else
      create_general_description_(description_);
    #endif
  }

  ~csrMatrixRef() {
    #if IS_CUSPARSE11_AVAILABLE()
      cusparseDestroySpMat(description_);
    #else
      cusparseDestroyMatDescr(description_);
    #endif
  }

  int size(int index) const {
    return size_.at(index);
  }
};

// csrOutput is used to represent the output for `CusparseMatrixMultiplyOp`
// Note that `csrOutput` is different from `csrMatrixRef` and the purpose
// of this was to have a materialized  version of a CSR matrix.
// Moreover this implements a RAII guard for a cusparse descriptor
struct csrOutput {
  Tensor csr_indices_{};
  Tensor csr_pointers_{};
  at::Tensor csr_values_{};
  int nnz_{0};
  std::vector<int> size_;

  cusparseMatDescr_t description_{0};

  csrOutput(const std::vector<int> &size) : size_{size} {
    create_general_description_(description_);
  }

  ~csrOutput() {
    cusparseDestroyMatDescr(description_);
  }

  int size(int index) const {
    return size_.at(index);
  }
};

#if IS_CUSPARSE11_AVAILABLE()

// RAII guard helps to support cuSparse 11 API for `A @ B` operation
// This generic template exists because with cuSparse the `scalar_t` type could be a double or float
template <class scalar_t>
struct CusparseMatrixMultiplyOp {

  cusparseSpGEMMDescr_t spgemmDesc;

  CusparseMatrixMultiplyOp() {
    static_assert(
      std::is_same<c10::Half, scalar_t>::value ||
          std::is_same<c10::BFloat16, scalar_t>::value ||
          std::is_same<float, scalar_t>::value ||
          std::is_same<double, scalar_t>::value ||
          std::is_same<c10::complex<float>, scalar_t>::value ||
          std::is_same<c10::complex<double>, scalar_t>::value,
      "cusparseSpGEMM only supports data type of half, bfloat16, float, double and complex float, double.");
    // SpGEMM Computation
    TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));
  }

  ~CusparseMatrixMultiplyOp() {
    // destroy matrix/vector descriptors
    cusparseSpGEMM_destroyDescr(spgemmDesc);
  }

  csrOutput operator ()(
      const csrMatrixRef<scalar_t>& A,
      const csrMatrixRef<scalar_t>& B,
      Tensor& output_values,
      Tensor& output_indices) {
    const int A_num_rows = A.size(0);

    const int B_num_cols = B.size(1);

    csrOutput out({A.size(0), B.size(1)});

    out.csr_pointers_ = at::empty({out.size(0) + 1}, output_indices.options().dtype(kInt));

    int* dC_csrOffsets = out.csr_pointers_.data_ptr<int>();
    int* dC_columns = nullptr;
    scalar_t* dC_values = nullptr;

    scalar_t alpha = 1.0f;
    scalar_t beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    csrMatrixRef<scalar_t> C(
      dC_columns,
      dC_csrOffsets,
      dC_values,
      /*nnz*/0,
      {A_num_rows, B_num_cols}
    );

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    cusparseSpMatDescr_t matA = A.description_;
    cusparseSpMatDescr_t matB = B.description_;
    cusparseSpMatDescr_t matC = C.description_;
    //--------------------------------------------------------------------------

    cudaDataType computeType = at::cuda::getCudaDataType<scalar_t>();

    // If a specific GPU model does not provide native support for a given data type,
    // the routine returns CUSPARSE_STATUS_ARCH_MISMATCH error
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major >= 5 && !((10*prop->major + prop->minor) < 53 && computeType == CUDA_R_16F),
        "sparse_mm: CUDA Float16 requires compute capability >= 53 (current: ", prop->major, prop->minor, ")");
    TORCH_CHECK(!(prop->major < 8 && computeType == CUDA_R_16BF),
        "sparse_mm: CUDA BFloat16 requires compute capability >= 80 (current: ", prop->major, prop->minor, ")");

    // ask bufferSize1 bytes for external memory
    TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle,
        opA,
        opB,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        computeType,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc,
        &bufferSize1,
        NULL));

    auto& allocator = *::c10::cuda::CUDACachingAllocator::get();

    at::DataPtr dataPtr1 = allocator.allocate(bufferSize1);
    dBuffer1 = dataPtr1.get();
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle,
        opA,
        opB,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        computeType,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc,
        &bufferSize1,
        dBuffer1));

    // ask bufferSize2 bytes for external memory
    TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_compute(
        handle,
        opA,
        opB,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        computeType,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc,
        &bufferSize2,
        NULL));

    at::DataPtr dataPtr2 = allocator.allocate(bufferSize2);
    dBuffer2 = dataPtr2.get();

    // compute the intermediate product of A * B
    TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_compute(
        handle,
        opA,
        opB,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        computeType,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc,
        &bufferSize2,
        dBuffer2));
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    TORCH_CUDASPARSE_CHECK(
        cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1));
    // allocate matrix C
    // allocate C offsets
    out.nnz_ = C_num_nnz1;

    out.csr_indices_ = at::empty({out.nnz_}, output_indices.options().dtype(kInt));
    out.csr_values_ = at::empty({out.nnz_}, output_values.options());
    dC_columns = out.csr_indices_.data_ptr<int>();
    dC_values = out.csr_values_.data_ptr<scalar_t>();

    // update matC with the new pointers
    TORCH_CUDASPARSE_CHECK(
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values));

    // copy the final products to the matrix C
    TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_copy(
        handle,
        opA,
        opB,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        computeType,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc));
    return out;
  }
};


template struct CusparseMatrixMultiplyOp<float>;

template struct CusparseMatrixMultiplyOp<double>;

#else // if not IS_CUSPARSE11_AVAILABLE()

using DcsrMatrixRef = csrMatrixRef<double>;
using ScsrMatrixRef = csrMatrixRef<float>;

// RAII guard helps to support cuSparse 10 API for `A @ B` operation
// This generic template exists because with cuSparse the `scalar_t` type could be a double or float
template <class scalar_t>
struct CusparseMatrixMultiplyOp {
  csrOutput operator()(
      const csrMatrixRef<scalar_t>& lhs,
      const csrMatrixRef<scalar_t>& rhs,
      Tensor &output_values,
      Tensor &output_indices)
  {
    static_assert(false&&sizeof(scalar_t), "cusparse csr sparse-sparse MM only supports data type of float and double.");
  }
};

// Specializacion for `A @ B` operation for double values with cuSparse
template<> struct CusparseMatrixMultiplyOp<double> {
  csrgemm2Info_t gemm2Info_;

  CusparseMatrixMultiplyOp() {
    TORCH_CUDASPARSE_CHECK(cusparseCreateCsrgemm2Info(&gemm2Info_));
  }
  ~CusparseMatrixMultiplyOp() {
    cusparseDestroyCsrgemm2Info(gemm2Info_);
  }

  csrOutput operator ()(
      const DcsrMatrixRef& lhs,
      const DcsrMatrixRef& rhs,
      Tensor &output_values,
      Tensor &output_indices) {
    double alpha = 1.0;
    DcsrMatrixRef empty;
    return Dgemm2(lhs, rhs, empty, &alpha, nullptr, output_values, output_indices);
  }

  csrOutput Dgemm2(
      const DcsrMatrixRef& A,
      const DcsrMatrixRef& B,
      const DcsrMatrixRef& C,
      const double* alpha,
      const double* beta,
      Tensor &output_values,
      Tensor &output_indices) {
    void* buffer_{nullptr};
    cusparseHandle_t cusparseHandle_ = at::cuda::getCurrentCUDASparseHandle();
    TORCH_CUDASPARSE_CHECK(cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST));

    csrOutput out({A.size(0), B.size(1)});
    int innerSize = confirm_mult_size(A.size_, B.size_);
    out.csr_pointers_ = at::empty({out.size(0) + 1}, output_indices.options().dtype(kInt));

    // Compute needed buffer size
    size_t new_bubber_sz;
    TORCH_CUDASPARSE_CHECK(cusparseDcsrgemm2_bufferSizeExt(
        cusparseHandle_,
        out.size(0),
        out.size(1),
        innerSize,
        alpha,
        A.description_,
        A.nnz_,
        A.csr_pointers_,
        A.csr_indices_,
        B.description_,
        B.nnz_,
        B.csr_pointers_,
        B.csr_indices_,
        beta,
        C.description_,
        C.nnz_,
        C.csr_pointers_,
        C.csr_indices_,
        gemm2Info_,
        &new_bubber_sz));

    // (Re)allocate buffer if needed
    auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
    at::DataPtr data_ptr = allocator.allocate(new_bubber_sz);
    buffer_ = data_ptr.get();

    // Find the resulting non-zero pattern.
    TORCH_CUDASPARSE_CHECK(cusparseXcsrgemm2Nnz(
        cusparseHandle_,
        out.size(0),
        out.size(1),
        innerSize,
        A.description_,
        A.nnz_,
        A.csr_pointers_,
        A.csr_indices_,
        B.description_,
        B.nnz_,
        B.csr_pointers_,
        B.csr_indices_,
        C.description_,
        C.nnz_,
        C.csr_pointers_,
        C.csr_indices_,
        out.description_,
        out.csr_pointers_.data_ptr<int>(),
        &out.nnz_,
        gemm2Info_,
        buffer_));

    out.csr_indices_ = at::empty({out.nnz_}, output_indices.options().dtype(kInt));
    out.csr_values_ = at::empty({out.nnz_}, output_values.options());

    // Perform the gemm2 operation for doubles
    // out = alpha ∗ A ∗ B + beta ∗ C
    TORCH_CUDASPARSE_CHECK(cusparseDcsrgemm2(
        cusparseHandle_,
        out.size(0),
        out.size(1),
        innerSize,
        alpha,
        A.description_,
        A.nnz_,
        A.csr_values_,
        A.csr_pointers_,
        A.csr_indices_,
        B.description_,
        B.nnz_,
        B.csr_values_,
        B.csr_pointers_,
        B.csr_indices_,
        beta,
        C.description_,
        C.nnz_,
        C.csr_values_,
        C.csr_pointers_,
        C.csr_indices_,
        out.description_,
        out.csr_values_.data_ptr<double>(),
        out.csr_pointers_.data_ptr<int>(),
        out.csr_indices_.data_ptr<int>(),
        gemm2Info_,
        buffer_));
    return out;
  }
};

// Specializacion for `A @ B` operation for float values with cuSparse
template<> struct CusparseMatrixMultiplyOp<float> {
  csrgemm2Info_t gemm2Info_;

  CusparseMatrixMultiplyOp() {
    TORCH_CUDASPARSE_CHECK(cusparseCreateCsrgemm2Info(&gemm2Info_));

  }
  ~CusparseMatrixMultiplyOp() {
    cusparseDestroyCsrgemm2Info(gemm2Info_);
  }
  csrOutput operator()(
      const ScsrMatrixRef& lhs,
      const ScsrMatrixRef& rhs,
      Tensor &output_values,
      Tensor &output_indices) {
    float alpha = 1.0;
    ScsrMatrixRef empty;
    return Sgemm2(lhs, rhs, empty, &alpha, nullptr, output_values, output_indices);
  }

  csrOutput Sgemm2(
      const ScsrMatrixRef& A,
      const ScsrMatrixRef& B,
      const ScsrMatrixRef& C,
      const float* alpha,
      const float* beta,
      Tensor &output_values,
      Tensor &output_indices) {
    void* buffer_{nullptr};
    cusparseHandle_t cusparseHandle_ = at::cuda::getCurrentCUDASparseHandle();
    TORCH_CUDASPARSE_CHECK(cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST));

    csrOutput out({A.size(0), B.size(1)});

    int innerSize = confirm_mult_size(A.size_, B.size_);

    out.csr_pointers_ = at::empty({out.size(0) + 1}, output_indices.options().dtype(kInt));

    // Compute needed buffer size
    size_t new_bubber_sz;
    TORCH_CUDASPARSE_CHECK(cusparseScsrgemm2_bufferSizeExt(
        cusparseHandle_,
        out.size(0),
        out.size(1),
        innerSize,
        alpha,
        A.description_,
        A.nnz_,
        A.csr_pointers_,
        A.csr_indices_,
        B.description_,
        B.nnz_,
        B.csr_pointers_,
        B.csr_indices_,
        beta,
        C.description_,
        C.nnz_,
        C.csr_pointers_,
        C.csr_indices_,
        gemm2Info_,
        &new_bubber_sz));

    auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
    at::DataPtr data_ptr = allocator.allocate(new_bubber_sz);
    buffer_ = data_ptr.get();

    // Find the resulting non-zero pattern.
    TORCH_CUDASPARSE_CHECK(cusparseXcsrgemm2Nnz(
        cusparseHandle_,
        out.size(0),
        out.size(1),
        innerSize,
        A.description_,
        A.nnz_,
        A.csr_pointers_,
        A.csr_indices_,
        B.description_,
        B.nnz_,
        B.csr_pointers_,
        B.csr_indices_,
        C.description_,
        C.nnz_,
        C.csr_pointers_,
        C.csr_indices_,
        out.description_,
        out.csr_pointers_.data_ptr<int>(),
        &out.nnz_,
        gemm2Info_,
        buffer_));

    out.csr_indices_ = at::empty({out.nnz_}, output_indices.options().dtype(kInt));
    out.csr_values_ = at::empty({out.nnz_}, output_values.options());

    // Perform the gemm2 operation for doubles
    // out = alpha ∗ A ∗ B + beta ∗ C
    TORCH_CUDASPARSE_CHECK(cusparseScsrgemm2(
        cusparseHandle_,
        out.size(0),
        out.size(1),
        innerSize,
        alpha,
        A.description_,
        A.nnz_,
        A.csr_values_,
        A.csr_pointers_,
        A.csr_indices_,
        B.description_,
        B.nnz_,
        B.csr_values_,
        B.csr_pointers_,
        B.csr_indices_,
        beta,
        C.description_,
        C.nnz_,
        C.csr_values_,
        C.csr_pointers_,
        C.csr_indices_,
        out.description_,
        out.csr_values_.data_ptr<float>(),
        out.csr_pointers_.data_ptr<int>(),
        out.csr_indices_.data_ptr<int>(),
        gemm2Info_,
        buffer_));
    return out;
  }
};



#endif // IS_CUSPARSE11_AVAILABLE()

template <typename scalar_t>
void sparse_sparse_matmul_cuda_kernel(
    Tensor& result,
    const Tensor& mat1,
    const Tensor& mat2) {

  static_assert(
    std::is_same<c10::Half, scalar_t>::value ||
        std::is_same<c10::BFloat16, scalar_t>::value ||
        std::is_same<float, scalar_t>::value ||
        std::is_same<double, scalar_t>::value ||
        std::is_same<c10::complex<float>, scalar_t>::value ||
        std::is_same<c10::complex<double>, scalar_t>::value,
    "sparse_sparse_matmul_cuda_kernel only supports data type of half, bfloat16, float, double and complex float, double.");

  // older versions of cusparse on Windows segfault for complex128 dtype
#if defined(_WIN32) && defined(CUSPARSE_VERSION) && CUSPARSE_VERSION < 11400
  TORCH_CHECK(
      !(mat1.scalar_type() == ScalarType::ComplexDouble),
      "Sparse multiplication with complex128 dtype inputs is not supported with current CUDA version. Please upgrade to CUDA Toolkit 11.2.1+");
#endif

  Tensor mat1_indices_ = mat1._indices().contiguous();
  Tensor mat1_values = mat1._values().contiguous();

  Tensor mat1_row_indices = mat1_indices_.select(0, 0);
  Tensor mat1_col_indices = mat1_indices_.select(0, 1);

  Tensor mat1_indptr = _to_csr_int(mat1_row_indices, mat1.size(0), mat1._nnz());

  Tensor mat1_indices = at::empty(
      {mat1_col_indices.size(0)}, mat1_col_indices.options().dtype(kInt));

  mat1_indices.copy_(mat1_col_indices);

  Tensor mat2_indices_ = mat2._indices().contiguous();
  Tensor mat2_values = mat2._values().contiguous();
  Tensor mat2_row_indices = mat2_indices_.select(0, 0);
  Tensor mat2_col_indices = mat2_indices_.select(0, 1);

  Tensor mat2_indptr = _to_csr_int(mat2_row_indices, mat2.size(0), mat2._nnz());
  Tensor mat2_indices = at::empty({mat2_col_indices.size(0)}, mat2_col_indices.options().dtype(kInt));
  mat2_indices.copy_(mat2_col_indices);

  auto m = mat1.size(0);
  auto k1 = mat1.size(1);

  auto k2 = mat2.size(0);
  auto n = mat2.size(1);
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k1 <= INT_MAX),
    "At the moment, cusparseDcsrgemm2 only supports m, n, k, nnz with the bound [val] <= ", INT_MAX, ".",
    "If you need this, please file an issue on GitHub."
  );
  auto output_indices = result._indices();
  auto output_values = result._values();

  if ((k1 == 0 && k2 == 0) || (n == 0 && m == 0)) {
    output_indices.zero_();
    output_values.zero_();
    return;
  }

  csrMatrixRef<scalar_t> csr_mat1(
      mat1_indices.data_ptr<int>(),
      mat1_indptr.data_ptr<int>(),
      mat1_values.data_ptr<scalar_t>(),
      (int)mat1._nnz(),
      {(int)mat1.size(0), (int)mat1.size(1)});

  csrMatrixRef<scalar_t> csr_mat2(
      mat2_indices.data_ptr<int>(),
      mat2_indptr.data_ptr<int>(),
      mat2_values.data_ptr<scalar_t>(),
      (int)mat2._nnz(),
      {(int)mat2.size(0), (int)mat2.size(1)});

  // Sparse matrix multiplication
  CusparseMatrixMultiplyOp<scalar_t> op;
  csrOutput csr_output = op(csr_mat1, csr_mat2, output_values, output_indices);
  auto nnz = csr_output.nnz_;

  output_values.set_(csr_output.csr_values_);
  output_indices.resize_({2, nnz});
  auto output_indices_accessor = output_indices.packed_accessor64<int64_t, 2>();

  auto csr_output_pointers_accessor =
      csr_output.csr_pointers_.packed_accessor64<int, 1>();

  auto csr_output_ind_accessor =
      csr_output.csr_indices_.packed_accessor64<int, 1>();

  auto major_dim = result.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::cuda::ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

  // Filling the COO row indices
  thrust::for_each(
      policy,
      thrust::make_counting_iterator(int64_t(0)),
      thrust::make_counting_iterator(int64_t(major_dim)),
      [output_indices_accessor,
       csr_output_pointers_accessor,
       major_dim,
       nnz] __device__(int64_t i) {
        auto Ap = csr_output_pointers_accessor.data();
        int64_t* indices_row = output_indices_accessor[0].data();

        for (int jj = Ap[i];  jj < Ap[i + 1]; jj++) {
          indices_row[jj] = i;
        }
      });

  // Filling the COO column indices
  thrust::for_each(
    policy,
    thrust::make_counting_iterator(int64_t(0)),
    thrust::make_counting_iterator(int64_t(csr_output.nnz_)),
    [output_indices_accessor,
      csr_output_pointers_accessor,
      csr_output_ind_accessor,
      major_dim,
      nnz] __device__(int64_t i) {
      int64_t* indices_col = output_indices_accessor[1].data();
      indices_col[i] = csr_output_ind_accessor[i];
    });
}

} // end anonymous namespace

Tensor sparse_sparse_matmul_cuda(const Tensor& mat1_, const Tensor& mat2_) {
  TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
  TORCH_INTERNAL_ASSERT(mat2_.is_sparse());
  TORCH_CHECK(mat1_.dim() == 2);
  TORCH_CHECK(mat2_.dim() == 2);
  TORCH_CHECK(mat1_.dense_dim() == 0, "sparse_mm: scalar values expected, mat1 got ", mat1_.dense_dim(), "D values");
  TORCH_CHECK(mat2_.dense_dim() == 0, "sparse_mm: scalar values expected, mat2 got ", mat2_.dense_dim(), "D values");

  TORCH_CHECK(
      mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

  TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
           "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

  auto output = at::native::empty_like(mat1_);
  output.sparse_resize_and_clear_({mat1_.size(0), mat2_.size(1)}, mat1_.sparse_dim(), 0);

#if IS_CUSPARSE11_AVAILABLE()
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, mat1_.scalar_type(), "sparse_matmul", [&] {
    sparse_sparse_matmul_cuda_kernel<scalar_t>(output, mat1_.coalesce(), mat2_.coalesce());
  });
#else
  AT_DISPATCH_FLOATING_TYPES(mat1_.scalar_type(), "sparse_matmul", [&] {
    sparse_sparse_matmul_cuda_kernel<scalar_t>(output, mat1_.coalesce(), mat2_.coalesce());
  });
#endif
  return output;
}

} // namespace at::native
