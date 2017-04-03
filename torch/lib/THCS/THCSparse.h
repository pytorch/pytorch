#ifndef THC_SPARSE_INC
#define THC_SPARSE_INC

#include <THC/THCGeneral.h>

THC_API void THCudaSparse_Xcoo2csr(THCState *state, const int *coorowind, long nnz, long m, int *csrrowptr);

/* Level 3 */
THC_API void THCudaSparse_Scsrmm2(THCState *state, char transa, char transb, long m, long n, long k, long nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, long ldb, float beta, float *c, long ldc);
THC_API void THCudaSparse_Dcsrmm2(THCState *state, char transa, char transb, long m, long n, long k, long nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, long ldb, double beta, double *c, long ldc);

/* format conversion */
THC_API void THCudaSparse_CreateIdentityPermutation(THCState *state, long nnz, int *P);
THC_API void THCudaSparse_Xcsrsort_bufferSizeExt(THCState *state, long m, long n, long nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes);
THC_API void THCudaSparse_Xcsrsort(THCState *state, long m, long n, long nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer);
THC_API void THCudaSparse_Xcoosort_bufferSizeExt(THCState *state, long m, long n, long nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes);
THC_API void THCudaSparse_XcoosortByRow(THCState *state, long m, long n, long nnz, int *cooRows, int *cooCols, int *P, void *pBuffer);

#endif
