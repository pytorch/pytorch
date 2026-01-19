#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cusparse.h>

#include <library_types.h>

namespace at::native::sparse::cuda {

void Xcoo2csr(const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  TORCH_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
    "cusparseXcoo2csr only supports m, nnz with the bound [val] <= ",
    INT_MAX);

  int i_nnz = static_cast<int>(nnz);
  int i_m = static_cast<int>(m);

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoo2csr(handle, coorowind, i_nnz, i_m, csrrowptr, CUSPARSE_INDEX_BASE_ZERO));
}

cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    TORCH_CHECK(false, "trans must be one of: t, n, c");
  }
}

namespace {
template<typename T>
void _csrmm2(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  T *alpha, T *csrvala, int *csrrowptra, int *csrcolinda,
  T *b, int64_t ldb, T *beta, T *c, int64_t ldc,
  cudaDataType cusparse_value_type)
{
  if (csrvala == nullptr || b == nullptr || c == nullptr) return;

  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  // cusparseSpMM actually supports int64_t.
  // In order to support int64 here, index pointers csrrowptra, csrcolinda have to be passed as int64_t.
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "At the moment, cusparseSpMM only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX, ".",
    "If you need this, please file an issue on GitHub."
  );

  int64_t ma = m, ka = k;
  if (transa != 'n') std::swap(ma, ka);

  cusparseSpMatDescr_t descA;
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
    &descA,                     /* output */
    ma, ka, nnz,                /* rows, cols, number of non zero elements */
    csrrowptra,                 /* row offsets of the sparse matrix, size = rows +1 */
    csrcolinda,                 /* column indices of the sparse matrix, size = nnz */
    csrvala,                    /* values of the sparse matrix, size = nnz */
    CUSPARSE_INDEX_32I,         /* data type of row offsets index */
    CUSPARSE_INDEX_32I,         /* data type of col indices */
    CUSPARSE_INDEX_BASE_ZERO,   /* base index of row offset and col index */
    cusparse_value_type         /* data type of values */
  ));

  int64_t kb = k, nb = n;
  if (transb != 'n') std::swap(kb, nb);

  cusparseDnMatDescr_t descB;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
    &descB,               /* output */
    kb, nb, ldb,          /* rows, cols, leading dimension */
    b,                    /* values */
    cusparse_value_type,  /* data type of values */
    CUSPARSE_ORDER_COL    /* memory layout, ONLY column-major is supported now */
  ));

  cusparseDnMatDescr_t descC;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
    &descC,               /* output */
    m, n, ldc,            /* rows, cols, leading dimension */
    c,                    /* values */
    cusparse_value_type,  /* data type of values */
    CUSPARSE_ORDER_COL    /* memory layout, ONLY column-major is supported now */
  ));


  auto handle = at::cuda::getCurrentCUDASparseHandle();
  // ALG1 is broken on SM89 as of CUDA 11.8+
#if !defined(USE_ROCM)
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  auto default_alg = prop->major == 8 && prop->minor == 9 ? CUSPARSE_SPMM_CSR_ALG2 : CUSPARSE_SPMM_CSR_ALG1;
#else
  auto default_alg = CUSPARSE_SPMM_CSR_ALG1;
#endif

  // cusparseSpMM_bufferSize returns the bufferSize that can be used by cusparseSpMM
  size_t bufferSize;
  TORCH_CUDASPARSE_CHECK(cusparseSpMM_bufferSize(
    handle, opa, opb,
    alpha,
    descA, descB,
    beta,
    descC,
    cusparse_value_type,      /* data type in which the computation is executed */
    default_alg,              /* default computing algorithm for CSR sparse matrix format */
    &bufferSize               /* output */
  ));

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(bufferSize);

  TORCH_CUDASPARSE_CHECK(cusparseSpMM(
    handle, opa, opb,
    alpha,
    descA, descB,
    beta,
    descC,
    cusparse_value_type,      /* data type in which the computation is executed */
    default_alg,              /* default computing algorithm for CSR sparse matrix format */
    dataPtr.get()             /* external buffer */
  ));

  TORCH_CUDASPARSE_CHECK(cusparseDestroySpMat(descA));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(descB));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(descC));

  // TODO: Proper fix is to create real descriptor classes
}
} // end anonymous namespace

template<typename T>
void csrmm2(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  T alpha, T *csrvala, int *csrrowptra, int *csrcolinda,
  T *b, int64_t ldb, T beta, T *c, int64_t ldc)
{
  static_assert(false&&sizeof(T), "cusparse csr MM only supports data type of float, double, cfloat and cdouble.");
}

template<> void csrmm2<float>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  float alpha, float *csrvala, int *csrrowptra, int *csrcolinda,
  float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  _csrmm2(transa, transb, m, n, k, nnz, &alpha, csrvala, csrrowptra, csrcolinda, b, ldb, &beta, c, ldc, CUDA_R_32F);
}

template<> void csrmm2<double>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  double alpha, double *csrvala, int *csrrowptra, int *csrcolinda,
  double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  _csrmm2(transa, transb, m, n, k, nnz, &alpha, csrvala, csrrowptra, csrcolinda, b, ldb, &beta, c, ldc, CUDA_R_64F);
}

template<> void csrmm2<c10::complex<float>>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  c10::complex<float> alpha, c10::complex<float> *csrvala, int *csrrowptra, int *csrcolinda,
  c10::complex<float> *b, int64_t ldb, c10::complex<float> beta, c10::complex<float> *c, int64_t ldc)
{
  _csrmm2(transa, transb, m, n, k, nnz,
    reinterpret_cast<cuComplex*>(&alpha),
    reinterpret_cast<cuComplex*>(csrvala),
    csrrowptra,
    csrcolinda,
    reinterpret_cast<cuComplex*>(b),
    ldb,
    reinterpret_cast<cuComplex*>(&beta),
    reinterpret_cast<cuComplex*>(c), ldc, CUDA_C_32F);
}

template<> void csrmm2<c10::complex<double>>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  c10::complex<double> alpha, c10::complex<double> *csrvala, int *csrrowptra, int *csrcolinda,
  c10::complex<double> *b, int64_t ldb, c10::complex<double> beta, c10::complex<double> *c, int64_t ldc)
{
  _csrmm2(transa, transb, m, n, k, nnz,
    reinterpret_cast<cuDoubleComplex*>(&alpha),
    reinterpret_cast<cuDoubleComplex*>(csrvala),
    csrrowptra,
    csrcolinda,
    reinterpret_cast<cuDoubleComplex*>(b),
    ldb,
    reinterpret_cast<cuDoubleComplex*>(&beta),
    reinterpret_cast<cuDoubleComplex*>(c), ldc, CUDA_C_64F);
}

/* format conversion */
void CreateIdentityPermutation(int64_t nnz, int *P) {
  TORCH_CHECK((nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_nnz = static_cast<int>(nnz);

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseCreateIdentityPermutation(handle, i_nnz, P);
}

void Xcsrsort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <=",
    INT_MAX);
  int i_m = static_cast<int>(m);
  int i_n = static_cast<int>(n);
  int i_nnz = static_cast<int>(nnz);

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcsrsort_bufferSizeExt(handle, i_m, i_n, i_nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

void Xcsrsort(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = static_cast<int>(m);
  int i_n = static_cast<int>(n);
  int i_nnz = static_cast<int>(nnz);

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  TORCH_CUDASPARSE_CHECK(cusparseXcsrsort(handle, i_m, i_n, i_nnz, desc, csrRowPtr, csrColInd, P, pBuffer));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

void Xcoosort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcoosort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = static_cast<int>(m);
  int i_n = static_cast<int>(n);
  int i_nnz = static_cast<int>(nnz);

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
}

void XcoosortByRow(int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "XcoosortByRow only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = static_cast<int>(m);
  int i_n = static_cast<int>(n);
  int i_nnz = static_cast<int>(nnz);

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows, cooCols, P, pBuffer));
}


} // namespace at::native::sparse::cuda
