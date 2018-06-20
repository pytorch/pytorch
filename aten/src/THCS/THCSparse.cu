#include "THCSparse.h"

void THCudaSparse_Xcoo2csr(THCState *state, const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

/* Level 3 */
void THCudaSparse_Scsrmm2(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCudaSparse_Dcsrmm2(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

/* format conversion */
void THCudaSparse_CreateIdentityPermutation(THCState *state, int64_t nnz, int *P) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCudaSparse_Xcsrsort_bufferSizeExt(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCudaSparse_Xcsrsort(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCudaSparse_Xcoosort_bufferSizeExt(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THCudaSparse_XcoosortByRow(THCState *state, int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}
