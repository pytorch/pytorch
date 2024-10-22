#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cusparse.h>

// LIMITATION (cusparseSpMM):
// The generic APIs are available on all platforms on CUDA 11.0
// For CUDA 10.1+ it is available for all platforms except Windows.
// Using these APIs in any other systems will result in compile-time or run-time failures.
// Their support will be extended in the next releases.

#if defined(CUDART_VERSION) && (CUSPARSE_VERSION >= 11000 || (!defined(_MSC_VER) && CUSPARSE_VERSION >= 10301))
#define IS_SPMM_AVAILABLE() 1
#else
#define IS_SPMM_AVAILABLE() 0
#endif

#if defined(USE_ROCM)
#define IS_SPMM_HIP_AVAILABLE() 1
#else
#define IS_SPMM_HIP_AVAILABLE() 0
#endif

#if IS_SPMM_AVAILABLE() || IS_SPMM_HIP_AVAILABLE()
#include <library_types.h>
#endif

#if !defined(CUSPARSE_VERSION) || (CUSPARSE_VERSION < 10100)
const char* cusparseGetErrorString(cusparseStatus_t status) {
  switch(status)
  {
    case CUSPARSE_STATUS_SUCCESS:
      return "success";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "library not initialized";

    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "resource allocation failed";

    case CUSPARSE_STATUS_INVALID_VALUE:
      return "an invalid numeric value was used as an argument";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "an absent device architectural feature is required";

    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "an access to GPU memory space failed";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "the GPU program failed to execute";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "an internal operation failed";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "the matrix type is not supported by this function";

    case CUSPARSE_STATUS_ZERO_PIVOT:
      return "an entry of the matrix is either structural zero or numerical zero (singular block)";

    default:
      return "unknown error";
  }
}
#endif

namespace at::native::sparse::cuda {

void Xcoo2csr(const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  TORCH_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
    "cusparseXcoo2csr only supports m, nnz with the bound [val] <= ",
    INT_MAX);

  int i_nnz = (int)nnz;
  int i_m = (int)m;

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

#if IS_SPMM_AVAILABLE() || IS_SPMM_HIP_AVAILABLE()

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
    CUSPARSE_INDEX_BASE_ZERO,   /* base index of row offset and col indes */
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
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // ALG1 is broken on SM89 as of CUDA 11.8+
#if !defined(USE_ROCM)
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

#else

void adjustLd(char transb, int64_t m, int64_t n, int64_t k, int64_t *ldb, int64_t *ldc)
{
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    *ldc = m;

  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

void Scsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, const float *alpha, const float *csrvala, int *csrrowptra, int *csrcolinda, const float *b, int64_t ldb, const float *beta, float *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseScsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  TORCH_CUDASPARSE_CHECK(cusparseScsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, beta, c, i_ldc));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

void Dcsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, const double *alpha, const double *csrvala, int *csrrowptra, int *csrcolinda, const double *b, int64_t ldb, const double *beta, double *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseDcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;


  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  TORCH_CUDASPARSE_CHECK(cusparseDcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, beta, c, i_ldc));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
  // TODO: Proper fix is to create real descriptor classes
}

template<class complex_target_t>
void Ccsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, const complex_target_t *alpha, const complex_target_t *csrvala, int *csrrowptra, int *csrcolinda, const complex_target_t *b, int64_t ldb, const complex_target_t *beta, complex_target_t *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseCcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  TORCH_CUDASPARSE_CHECK(cusparseCcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, beta, c, i_ldc));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

template<class complex_target_t>
void Zcsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, const complex_target_t *alpha, const complex_target_t *csrvala, int *csrrowptra, int *csrcolinda, const complex_target_t *b, int64_t ldb, const complex_target_t *beta, complex_target_t *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseZcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;


  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  TORCH_CUDASPARSE_CHECK(cusparseZcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, beta, c, i_ldc));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

// T can only be float or double
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
  Scsrmm2(transa, transb, m, n, k, nnz, &alpha, csrvala, csrrowptra, csrcolinda, b, ldb, &beta, c, ldc);
}

template<> void csrmm2<double>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  double alpha, double *csrvala, int *csrrowptra, int *csrcolinda,
  double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  Dcsrmm2(transa, transb, m, n, k, nnz, &alpha, csrvala, csrrowptra, csrcolinda, b, ldb, &beta, c, ldc);
}

template<> void csrmm2<c10::complex<float>>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  c10::complex<float> alpha, c10::complex<float> *csrvala, int *csrrowptra, int *csrcolinda,
  c10::complex<float> *b, int64_t ldb, c10::complex<float> beta, c10::complex<float> *c, int64_t ldc)
{

  #ifdef USE_ROCM
  Ccsrmm2(transa, transb, m, n, k, nnz,
    reinterpret_cast<const hipComplex*>(&alpha),
    reinterpret_cast<const hipComplex*>(csrvala),
    csrrowptra,
    csrcolinda,
    reinterpret_cast<const hipComplex*>(b),
    ldb,
    reinterpret_cast<const hipComplex*>(&beta),
    reinterpret_cast<hipComplex*>(c), ldc);
  #else
  Ccsrmm2(transa, transb, m, n, k, nnz,
      reinterpret_cast<const cuComplex*>(&alpha),
      reinterpret_cast<const cuComplex*>(csrvala),
      csrrowptra,
      csrcolinda,
      reinterpret_cast<const cuComplex*>(b),
      ldb,
      reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc);
  #endif
}

template<> void csrmm2<c10::complex<double>>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  c10::complex<double> alpha, c10::complex<double> *csrvala, int *csrrowptra, int *csrcolinda,
  c10::complex<double> *b, int64_t ldb, c10::complex<double> beta, c10::complex<double> *c, int64_t ldc)
{
  #ifdef USE_ROCM
  Zcsrmm2(transa, transb, m, n, k, nnz,
    reinterpret_cast<const hipDoubleComplex*>(&alpha),
    reinterpret_cast<const hipDoubleComplex*>(csrvala),
    csrrowptra,
    csrcolinda,
    reinterpret_cast<const hipDoubleComplex*>(b),
    ldb,
    reinterpret_cast<const hipDoubleComplex*>(&beta),
    reinterpret_cast<hipDoubleComplex*>(c), ldc);
  #else
  Zcsrmm2(transa, transb, m, n, k, nnz,
    reinterpret_cast<const cuDoubleComplex*>(&alpha),
    reinterpret_cast<const cuDoubleComplex*>(csrvala),
    csrrowptra,
    csrcolinda,
    reinterpret_cast<const cuDoubleComplex*>(b),
    ldb,
    reinterpret_cast<const cuDoubleComplex*>(&beta),
    reinterpret_cast<cuDoubleComplex*>(c), ldc);
  #endif
}


#endif

/* format conversion */
void CreateIdentityPermutation(int64_t nnz, int *P) {
  TORCH_CHECK((nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseCreateIdentityPermutation(handle, i_nnz, P);
}

void Xcsrsort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <=",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcsrsort_bufferSizeExt(handle, i_m, i_n, i_nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

void Xcsrsort(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

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
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
}

void XcoosortByRow(int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "XcoosortByRow only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows, cooCols, P, pBuffer));
}


} // namespace at::native::sparse::cuda
