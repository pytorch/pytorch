#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THLapack.cpp"
#else


TH_EXTERNC void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
TH_EXTERNC void sgels_(char *trans, int *m, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, float *work, int *lwork, int *info);
TH_EXTERNC void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double* vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
TH_EXTERNC void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda, float *wr, float *wi, float* vl, int *ldvl, float *vr, int *ldvr, float *work, int *lwork, int *info);
TH_EXTERNC void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
TH_EXTERNC void sgetrs_(char *trans, int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);
TH_EXTERNC void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);
TH_EXTERNC void spotri_(char *uplo, int *n, float *a, int *lda, int *info);
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

/* Compute for an N-by-N real nonsymmetric matrix A, the eigenvalues and,
optionally, the left and/or right eigenvectors */
void THLapack_(geev)(char jobvl, char jobvr, int n, scalar_t *a, int lda, scalar_t *wr, scalar_t *wi, scalar_t* vl, int ldvl, scalar_t *vr, int ldvr, scalar_t *work, int lwork, int *info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
#else
  sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
#endif
#else
  THError("geev : Lapack library not found in compile time\n");
#endif
}

void THLapack_(getrs)(char trans, int n, int nrhs, scalar_t *a, int lda, int *ipiv, scalar_t *b, int ldb, int *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  sgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#endif
#else
  THError("getrs : Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization based Matrix Inverse */
void THLapack_(potri)(char uplo, int n, scalar_t *a, int lda, int *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dpotri_(&uplo, &n, a, &lda, info);
#else
  spotri_(&uplo, &n, a, &lda, info);
#endif
#else
  THError("potri: Lapack library not found in compile time\n");
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

/* Build Q from output of geqrf */
void THLapack_(orgqr)(int m, int n, int k, scalar_t *a, int lda, scalar_t *tau, scalar_t *work, int lwork, int *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
#else
  sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
#endif
#else
  THError("orgqr: Lapack library not found in compile time\n");
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
