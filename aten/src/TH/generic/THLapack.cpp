#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THLapack.cpp"
#else

TH_EXTERNC void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
TH_EXTERNC void sgels_(char *trans, int *m, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, float *work, int *lwork, int *info);
TH_EXTERNC void sormqr_(char *side, char *trans, int *m, int *n, int *k, float *a, int *lda, float *tau, float *c, int *ldc, float *work, int *lwork, int *info);
TH_EXTERNC void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);


/* Solve overdetermined or underdetermined real linear systems involving an
M-by-N matrix A, or its transpose, using a QR or LQ factorization of A */
void THLapack_(gels)(char trans, int m, int n, int nrhs, scalar_t *a, int lda, scalar_t *b, int ldb, scalar_t *work, int lwork, int *info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#else
  sgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#endif
#else
  THError("gels : Lapack library not found in compile time\n");
#endif
}

#endif
