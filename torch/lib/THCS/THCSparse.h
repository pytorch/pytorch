#ifndef THC_SPARSE_INC
#define THC_SPARSE_INC

#ifdef THCS_EXPORTS
#define TH_EXPORTS
#endif

#include <THC/THCGeneral.h>

TH_API void THCudaSparse_Xcoo2csr(THCState *state, const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr);

/* Level 3 */
TH_API void THCudaSparse_Scsrmm2(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc);
TH_API void THCudaSparse_Dcsrmm2(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc);

/* format conversion */
TH_API void THCudaSparse_CreateIdentityPermutation(THCState *state, int64_t nnz, int *P);
TH_API void THCudaSparse_Xcsrsort_bufferSizeExt(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes);
TH_API void THCudaSparse_Xcsrsort(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer);
TH_API void THCudaSparse_Xcoosort_bufferSizeExt(THCState *state, int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes);
TH_API void THCudaSparse_XcoosortByRow(THCState *state, int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer);

#endif
