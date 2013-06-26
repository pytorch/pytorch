#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLapack.c"
#else


TH_EXTERNC void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
TH_EXTERNC void sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);
TH_EXTERNC void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
TH_EXTERNC void sgels_(char *trans, int *m, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, float *work, int *lwork, int *info);
TH_EXTERNC void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *info);
TH_EXTERNC void ssyev_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w, float *work, int *lwork, int *info);
TH_EXTERNC void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double* vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
TH_EXTERNC void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda, float *wr, float *wi, float* vl, int *ldvl, float *vr, int *ldvr, float *work, int *lwork, int *info);
TH_EXTERNC void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *info);
TH_EXTERNC void sgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *lda, float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *info);
TH_EXTERNC void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
TH_EXTERNC void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
TH_EXTERNC void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
TH_EXTERNC void sgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info);
  

void THLapack_(gesv)(int n, int nrhs, real *a, int lda, int *ipiv, real *b, int ldb, int* info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#endif
#else
  THError("gesv : Lapack library not found in compile time\n");
#endif
  return;
}

void THLapack_(gels)(char trans, int m, int n, int nrhs, real *a, int lda, real *b, int ldb, real *work, int lwork, int *info)
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

void THLapack_(syev)(char jobz, char uplo, int n, real *a, int lda, real *w, real *work, int lwork, int *info)
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

void THLapack_(geev)(char jobvl, char jobvr, int n, real *a, int lda, real *wr, real *wi, real* vl, int ldvl, real *vr, int ldvr, real *work, int lwork, int *info)
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

void THLapack_(gesvd)(char jobu, char jobvt, int m, int n, real *a, int lda, real *s, real *u, int ldu, real *vt, int ldvt, real *work, int lwork, int *info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgesvd_( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#else
  sgesvd_( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#endif
#else
  THError("gesvd : Lapack library not found in compile time\n");
#endif
}

/* LU decomposition */
void THLapack_(getrf)(int m, int n, real *a, int lda, int *ipiv, int *info)
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
/* Matrix Inverse */
void THLapack_(getri)(int n, real *a, int lda, int *ipiv, real *work, int lwork, int* info)
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

#endif
