#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THLapack.cpp"
#else


TH_EXTERNC void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
TH_EXTERNC void strtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info);
TH_EXTERNC void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
TH_EXTERNC void sgels_(char *trans, int *m, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, float *work, int *lwork, int *info);
TH_EXTERNC void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *info);
TH_EXTERNC void ssyev_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w, float *work, int *lwork, int *info);
TH_EXTERNC void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double* vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
TH_EXTERNC void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda, float *wr, float *wi, float* vl, int *ldvl, float *vr, int *ldvr, float *work, int *lwork, int *info);
TH_EXTERNC void dgesdd_(char *jobz, int *m, int *n, double *a, int *lda, double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *iwork, int *info);
TH_EXTERNC void sgesdd_(char *jobz, int *m, int *n, float *a, int *lda, float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *iwork, int *info);
TH_EXTERNC void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
TH_EXTERNC void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
TH_EXTERNC void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
TH_EXTERNC void sgetrs_(char *trans, int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);
TH_EXTERNC void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
TH_EXTERNC void sgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info);
TH_EXTERNC void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);
TH_EXTERNC void spotri_(char *uplo, int *n, float *a, int *lda, int *info);
TH_EXTERNC void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
TH_EXTERNC void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
TH_EXTERNC void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
TH_EXTERNC void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
TH_EXTERNC void sormqr_(char *side, char *trans, int *m, int *n, int *k, float *a, int *lda, float *tau, float *c, int *ldc, float *work, int *lwork, int *info);
TH_EXTERNC void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);
TH_EXTERNC void spstrf_(char *uplo, int *n, float *a, int *lda, int *piv, int *rank, float *tol, float *work, int *info);
TH_EXTERNC void dpstrf_(char *uplo, int *n, double *a, int *lda, int *piv, int *rank, double *tol, double *work, int *info);


/* Solve a triangular system of the form A * X = B  or A^T * X = B */
void THLapack_(trtrs)(char uplo, char trans, char diag, int n, int nrhs, scalar_t *a, int lda, scalar_t *b, int ldb, int* info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, info);
#else
  strtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, info);
#endif
#else
  THError("trtrs : Lapack library not found in compile time\n");
#endif
  return;
}

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

/* Compute all eigenvalues and, optionally, eigenvectors of a real symmetric
matrix A */
void THLapack_(syev)(char jobz, char uplo, int n, scalar_t *a, int lda, scalar_t *w, scalar_t *work, int lwork, int *info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#else
  ssyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#endif
#else
  THError("syev : Lapack library not found in compile time\n");
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

/* Compute the singular value decomposition (SVD) of a real M-by-N matrix A,
optionally computing the left and/or right singular vectors */
void THLapack_(gesdd)(char jobz, int m, int n, scalar_t *a, int lda, scalar_t *s, scalar_t *u, int ldu, scalar_t *vt, int ldvt, scalar_t *work, int lwork, int *iwork, int *info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgesdd_( &jobz,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  iwork,  info);
#else
  sgesdd_( &jobz,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  iwork,  info);
#endif
#else
  THError("gesdd : Lapack library not found in compile time\n");
#endif
}

/* LU decomposition */
void THLapack_(getrf)(int m, int n, scalar_t *a, int lda, int *ipiv, int *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgetrf_(&m, &n, a, &lda, ipiv, info);
#else
  sgetrf_(&m, &n, a, &lda, ipiv, info);
#endif
#else
  THError("getrf : Lapack library not found in compile time\n");
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

/* Matrix Inverse */
void THLapack_(getri)(int n, scalar_t *a, int lda, int *ipiv, scalar_t *work, int lwork, int* info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgetri_(&n, a, &lda, ipiv, work, &lwork, info);
#else
  sgetri_(&n, a, &lda, ipiv, work, &lwork, info);
#endif
#else
  THError("getri : Lapack library not found in compile time\n");
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

/* Cholesky factorization with complete pivoting */
void THLapack_(pstrf)(char uplo, int n, scalar_t *a, int lda, int *piv, int *rank, scalar_t tol, scalar_t *work, int *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dpstrf_(&uplo, &n, a, &lda, piv, rank, &tol, work, info);
#else
  spstrf_(&uplo, &n, a, &lda, piv, rank, &tol, work, info);
#endif
#else
  THError("pstrf: Lapack library not found at compile time\n");
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
