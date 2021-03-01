#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THLapack.cpp"
#else

TH_EXTERNC void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
TH_EXTERNC void sgels_(char *trans, int *m, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, float *work, int *lwork, int *info);
TH_EXTERNC void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
TH_EXTERNC void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
TH_EXTERNC void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
TH_EXTERNC void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
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

/* QR decomposition */
void THLapack_(geqrf)(int m, int n, scalar_t *a, int lda, scalar_t *tau, scalar_t *work, int lwork, int *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
#else
  sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
#endif
#else
  THError("geqrf: Lapack library not found in compile time\n");
#endif
}

/* Multiply Q with a matrix using the output of geqrf */
void THLapack_(ormqr)(char side, char trans, int m, int n, int k, scalar_t *a, int lda, scalar_t *tau, scalar_t *c, int ldc, scalar_t *work, int lwork, int *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
#else
  sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
#endif
#else
  THError("ormqr: Lapack library not found in compile time\n");
#endif
}


#endif
