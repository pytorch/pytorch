#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLapack.c"
#else

void THLapack_(gesv)(int n, int nrhs, real *a, int lda, int *ipiv, real *b, int ldb, int* info)
{
#ifdef __LAPACK__
#if defined(TH_REAL_IS_DOUBLE)
  extern void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  extern void sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);
  sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#endif
#else
  THError("gesv : Lapack library not found in compile time\n");
#endif
  return;
}

void THLapack_(gels)(char trans, int m, int n, int nrhs, real *a, int lda, real *b, int ldb, real *work, int lwork, int *info)
{
#ifdef __LAPACK__
#if defined(TH_REAL_IS_DOUBLE)
  extern void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
  dgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#else
  extern void sgels_(char *trans, int *m, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, float *work, int *lwork, int *info);
  sgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#endif
#else
  THError("gels : Lapack library not found in compile time\n");
#endif
}

void THLapack_(syev)(char jobz, char uplo, int n, real *a, int lda, real *w, real *work, int lwork, int *info)
{
#ifdef __LAPACK__
#if defined(TH_REAL_IS_DOUBLE)
  extern void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *info);
  dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#else
  extern void ssyev_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w, float *work, int *lwork, int *info);
  ssyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#endif
#else
  THError("syev : Lapack library not found in compile time\n");
#endif
}

void THLapack_(gesvd)(char jobu, char jobvt, int m, int n, real *a, int lda, real *s, real *u, int ldu, real *vt, int ldvt, real *work, int lwork, int *info)
{
#ifdef __LAPACK__
#if defined(TH_REAL_IS_DOUBLE)
  extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *info);
  dgesvd_( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#else
  extern void sgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *lda, float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *info);
  sgesvd_( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#endif
#else
  THError("gesvd : Lapack library not found in compile time\n");
#endif
}

#endif
