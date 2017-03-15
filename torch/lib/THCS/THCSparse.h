#ifndef THC_SPARSE_INC
#define THC_SPARSE_INC

#include <THC/THCGeneral.h>

THC_API void THCudaSparse_Xcoo2csr(THCState *state, const int *coorowind, long nnz, long m, int *csrrowptr);

/* Level 3 */
THC_API void THCudaSparse_Scsrmm2(THCState *state, char transa, char transb, long m, long n, long k, long nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, long ldb, float beta, float *c, long ldc);
THC_API void THCudaSparse_Dcsrmm2(THCState *state, char transa, char transb, long m, long n, long k, long nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, long ldb, double beta, double *c, long ldc);

#endif
