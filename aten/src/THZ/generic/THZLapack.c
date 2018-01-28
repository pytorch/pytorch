#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZLapack.c"
#else

#include "THZTypeMacros.h"

// extern api declaration (floating points only)

TH_EXTERNC void THZ_LAPACK_NAME(gesv)(int *n, int *nrhs, ntype *a, int *lda, int *ipiv, ntype *b, int *ldb, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(trtrs)(char *uplo, char *trans, char *diag, int *n, int *nrhs, ntype *a, int *lda, ntype *b, int *ldb, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(gels)(char *trans, int *m, int *n, int *nrhs, ntype *a, int *lda, ntype *b, int *ldb, ntype *work, int *lwork, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(geev)(char *jobvl, char *jobvr, int *n, ntype *a, int *lda, ntype *wr, ntype *wi, ntype* vl, int *ldvl, ntype *vr, int *ldvr, ntype *work, int *lwork, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(gesvd)(char *jobu, char *jobvt, int *m, int *n, ntype *a, int *lda, ntype *s, ntype *u, int *ldu, ntype *vt, int *ldvt, ntype *work, int *lwork, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(getrf)(int *m, int *n, ntype *a, int *lda, int *ipiv, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(getrs)(char *trans, int *n, int *nrhs, ntype *a, int *lda, int *ipiv, ntype *b, int *ldb, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(getri)(int *n, ntype *a, int *lda, int *ipiv, ntype *work, int *lwork, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(potrf)(char *uplo, int *n, ntype *a, int *lda, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(potri)(char *uplo, int *n, ntype *a, int *lda, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(potrs)(char *uplo, int *n, int *nrhs, ntype *a, int *lda, ntype *b, int *ldb, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(geqrf)(int *m, int *n, ntype *a, int *lda, ntype *tau, ntype *work, int *lwork, int *info);
TH_EXTERNC void THZ_LAPACK_NAME(pstrf)(char *uplo, int *n, ntype *a, int *lda, int *piv, int *rank, ntype *tol, ntype *work, int *info);


#ifdef THZ_NTYPE_IS_COMPLEX
TH_EXTERNC void THZ_LAPACK_NAME(heev)(char *jobz, char *uplo, int *n, ntype *a, int *lda, ntype *w, ntype *work, int *lwork, int *info);
#else
TH_EXTERNC void THZ_LAPACK_NAME(syev)(char *jobz, char *uplo, int *n, ntype *a, int *lda, ntype *w, ntype *work, int *lwork, int *info);
#endif

#ifdef THZ_NTYPE_IS_COMPLEX
TH_EXTERNC void THZ_LAPACK_NAME(ungqr)(int *m, int *n, int *k, ntype *a, int *lda, ntype *tau, ntype *work, int *lwork, int *info);
#else
TH_EXTERNC void THZ_LAPACK_NAME(orgqr)(int *m, int *n, int *k, ntype *a, int *lda, ntype *tau, ntype *work, int *lwork, int *info);
#endif

#ifdef THZ_NTYPE_IS_COMPLEX
TH_EXTERNC void THZ_LAPACK_NAME(unmqr)(char *side, char *trans, int *m, int *n, int *k, ntype *a, int *lda, ntype *tau, ntype *c, int *ldc, ntype *work, int *lwork, int *info);
#else
TH_EXTERNC void THZ_LAPACK_NAME(ormqr)(char *side, char *trans, int *m, int *n, int *k, ntype *a, int *lda, ntype *tau, ntype *c, int *ldc, ntype *work, int *lwork, int *info);
#endif
// end of extern api declaration

/* Compute the solution to a ntype system of linear equations  A * X = B */
void THZLapack_(gesv)(int n, int nrhs, ntype *a, int lda, int *ipiv, ntype *b, int ldb, int* info)
{
#ifdef USE_LAPACK
  THZ_LAPACK_NAME(gesv)(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  THError("gesv : Lapack library not found in compile time\n");
#endif
  return;
}

/* Solve a triangular system of the form A * X = B  or A^T * X = B */
void THZLapack_(trtrs)(char uplo, char trans, char diag, int n, int nrhs, ntype *a, int lda, ntype *b, int ldb, int* info)
{
#ifdef USE_LAPACK
  THZ_LAPACK_NAME(trtrs)(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, info);
#else
  THError("trtrs : Lapack library not found in compile time\n");
#endif
  return;
}

/* Solve overdetermined or underdetermined ntype linear systems involving an
M-by-N matrix A, or its transpose, using a QR or LQ factorization of A */
void THZLapack_(gels)(char trans, int m, int n, int nrhs, ntype *a, int lda, ntype *b, int ldb, ntype *work, int lwork, int *info)
{
#ifdef USE_LAPACK
  THZ_LAPACK_NAME(gels)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#else
  THError("gels : Lapack library not found in compile time\n");
#endif
}

/* Compute all eigenvalues and, optionally, eigenvectors of a ntype symmetric
matrix A */
void THZLapack_(syev)(char jobz, char uplo, int n, ntype *a, int lda, ntype *w, ntype *work, int lwork, int *info)
{
#ifdef USE_LAPACK
  THZ_LAPACK_NAME(heev)(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#else
  THError("syev : Lapack library not found in compile time\n");
#endif
}

/* Compute for an N-by-N ntype nonsymmetric matrix A, the eigenvalues and,
optionally, the left and/or right eigenvectors */
void THZLapack_(geev)(char jobvl, char jobvr, int n, ntype *a, int lda, ntype *wr, ntype *wi, ntype* vl, int ldvl, ntype *vr, int ldvr, ntype *work, int lwork, int *info)
{
#ifdef USE_LAPACK
  THZ_LAPACK_NAME(geev)(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
#else
  THError("geev : Lapack library not found in compile time\n");
#endif
}

/* Compute the singular value decomposition (SVD) of a ntype M-by-N matrix A,
optionally computing the left and/or right singular vectors */
void THZLapack_(gesvd)(char jobu, char jobvt, int m, int n, ntype *a, int lda, ntype *s, ntype *u, int ldu, ntype *vt, int ldvt, ntype *work, int lwork, int *info)
{
#ifdef USE_LAPACK
  THZ_LAPACK_NAME(gesvd)( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#else
  THError("gesvd : Lapack library not found in compile time\n");
#endif
}

/* LU decomposition */
void THZLapack_(getrf)(int m, int n, ntype *a, int lda, int *ipiv, int *info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(getrf)(&m, &n, a, &lda, ipiv, info);
#else
  THError("getrf : Lapack library not found in compile time\n");
#endif
}

void THZLapack_(getrs)(char trans, int n, int nrhs, ntype *a, int lda, int *ipiv, ntype *b, int ldb, int *info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(getrs)(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  THError("getrs : Lapack library not found in compile time\n");
#endif
}

/* Matrix Inverse */
void THZLapack_(getri)(int n, ntype *a, int lda, int *ipiv, ntype *work, int lwork, int* info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(getri)(&n, a, &lda, ipiv, work, &lwork, info);
#else
  THError("getri : Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization */
void THZLapack_(potrf)(char uplo, int n, ntype *a, int lda, int *info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(potrf)(&uplo, &n, a, &lda, info);
#else
  THError("potrf : Lapack library not found in compile time\n");
#endif
}

/* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
void THZLapack_(potrs)(char uplo, int n, int nrhs, ntype *a, int lda, ntype *b, int ldb, int *info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(potrs)(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
#else
  THError("potrs: Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization based Matrix Inverse */
void THZLapack_(potri)(char uplo, int n, ntype *a, int lda, int *info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(potri)(&uplo, &n, a, &lda, info);
#else
  THError("potri: Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization with complete pivoting */
void THZLapack_(pstrf)(char uplo, int n, ntype *a, int lda, int *piv, int *rank, ntype tol, ntype *work, int *info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(pstrf)(&uplo, &n, a, &lda, piv, rank, &tol, work, info);
#else
  THError("pstrf: Lapack library not found at compile time\n");
#endif
}

/* QR decomposition */
void THZLapack_(geqrf)(int m, int n, ntype *a, int lda, ntype *tau, ntype *work, int lwork, int *info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(geqrf)(&m, &n, a, &lda, tau, work, &lwork, info);
#else
  THError("geqrf: Lapack library not found in compile time\n");
#endif
}

/* Build Q from output of geqrf */
void THZLapack_(orgqr)(int m, int n, int k, ntype *a, int lda, ntype *tau, ntype *work, int lwork, int *info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(ungqr)(&m, &n, &k, a, &lda, tau, work, &lwork, info);
#else
  THError("orgqr: Lapack library not found in compile time\n");
#endif
}

/* Multiply Q with a matrix using the output of geqrf */
void THZLapack_(ormqr)(char side, char trans, int m, int n, int k, ntype *a, int lda, ntype *tau, ntype *c, int ldc, ntype *work, int lwork, int *info)
{
#ifdef  USE_LAPACK
  THZ_LAPACK_NAME(unmqr)(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
#else
  THError("ormqr: Lapack library not found in compile time\n");
#endif
}


#endif
