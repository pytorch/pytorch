#include "THCSparse.h"

void THCudaSparse_Xcoo2csr(THCState *state, const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  THAssertMsg((m <= INT_MAX) && (nnz <= INT_MAX),
    "cusparseXcoo2csr only supports m, nnz with the bound [val] <= %d",
    INT_MAX);
  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  THCusparseCheck(cusparseXcoo2csr(handle, coorowind, nnz, m, csrrowptr,
    TH_INDEX_BASE ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO
  ));
}

cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    THError("trans must be one of: t, n, c");
    return CUSPARSE_OPERATION_TRANSPOSE;
  }
}

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

/* Level 3 */
void THCudaSparse_Scsrmm2(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseScsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&desc, CUSPARSE_INDEX_BASE_ONE);
#endif
  THCusparseCheck(cusparseScsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
}

void THCudaSparse_Dcsrmm2(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseDcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&desc, CUSPARSE_INDEX_BASE_ONE);
#endif
  THCusparseCheck(cusparseDcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
}

/* format conversion */
void THCudaSparse_CreateIdentityPermutation(THCState *state, int64_t nnz, int *P) {
  THAssertMsg((nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  cusparseCreateIdentityPermutation(handle, i_nnz, P);
}

void THCudaSparse_Xcsrsort_bufferSizeExt(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  THCusparseCheck(cusparseXcsrsort_bufferSizeExt(handle, i_m, i_n, i_nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

void THCudaSparse_Xcsrsort(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  cusparseSetMatIndexBase(&desc, CUSPARSE_INDEX_BASE_ONE);
#endif
  THCusparseCheck(cusparseXcsrsort(handle, i_m, i_n, i_nnz, desc, csrRowPtr, csrColInd, P, pBuffer));
}

void THCudaSparse_Xcoosort_bufferSizeExt(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
{
  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcoosort_bufferSizeExt only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  THCusparseCheck(cusparseXcoosort_bufferSizeExt(handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
}

void THCudaSparse_XcoosortByRow(THCState *state, int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  THAssertMsg((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "XcoosortByRow only supports m, n, nnz with the bound [val] <= %d",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseSetStream(handle, THCState_getCurrentStream(state));
  THCusparseCheck(cusparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows, cooCols, P, pBuffer));
}
