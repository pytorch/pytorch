#include "THCSparse.h"

void THCudaSparse_Xcoo2csr(THCState *state, const int *coorowind, long nnz, long m, int *csrrowptr) {
  if ((m <= INT_MAX) && (nnz <= INT_MAX))
  {
    cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
    cusparseSetStream(handle, THCState_getCurrentStream(state));
    THCusparseCheck(cusparseXcoo2csr(handle, coorowind, nnz, m, csrrowptr,
      TH_INDEX_BASE ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO
    ));
    return;
  }
  THError("cusparseXcoo2csr only supports m, nnz "
          "with the bound [val] <= %d", INT_MAX);
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

void adjustLd(char transb, long m, long n, long k, long *ldb, long *ldc)
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
void THCudaSparse_Scsrmm2(THCState *state, char transa, char transb, long m, long n, long k, long nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, long ldb, float beta, float *c, long ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
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
    return;
  }
  THError("cusparseScsrmm2 only supports m, n, k, nnz, ldb, ldc "
          "with the bound [val] <= %d", INT_MAX);
}

void THCudaSparse_Dcsrmm2(THCState *state, char transa, char transb, long m, long n, long k, long nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, long ldb, double beta, double *c, long ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
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
    return;
  }
  THError("cusparseDcsrmm2 only supports m, n, k, nnz, ldb, ldc "
          "with the bound [val] <= %d", INT_MAX);
}
