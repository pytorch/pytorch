#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZLapack.h"
#else

/* AX=B */
TH_API void THZLapack_(gesv)(int n, int nrhs, ntype *a, int lda, int *ipiv, ntype *b, int ldb, int* info);
/* Solve a triangular system of the form A * X = B  or A^T * X = B */
TH_API void THZLapack_(trtrs)(char uplo, char trans, char diag, int n, int nrhs, ntype *a, int lda, ntype *b, int ldb, int* info);
/* ||AX-B|| */
TH_API void THZLapack_(gels)(char trans, int m, int n, int nrhs, ntype *a, int lda, ntype *b, int ldb, ntype *work, int lwork, int *info);
/* Eigenvals */
TH_API void THZLapack_(syev)(char jobz, char uplo, int n, ntype *a, int lda, ntype *w, ntype *work, int lwork, int *info);
/* Non-sym eigenvals */
TH_API void THZLapack_(geev)(char jobvl, char jobvr, int n, ntype *a, int lda, ntype *wr, ntype *wi, ntype* vl, int ldvl, ntype *vr, int ldvr, ntype *work, int lwork, int *info);
/* svd */
TH_API void THZLapack_(gesvd)(char jobu, char jobvt, int m, int n, ntype *a, int lda, ntype *s, ntype *u, int ldu, ntype *vt, int ldvt, ntype *work, int lwork, int *info);
/* LU decomposition */
TH_API void THZLapack_(getrf)(int m, int n, ntype *a, int lda, int *ipiv, int *info);
TH_API void THZLapack_(getrs)(char trans, int n, int nrhs, ntype *a, int lda, int *ipiv, ntype *b, int ldb, int *info);
/* Matrix Inverse */
TH_API void THZLapack_(getri)(int n, ntype *a, int lda, int *ipiv, ntype *work, int lwork, int* info);

/* Positive Definite matrices */
/* Cholesky factorization */
void THZLapack_(potrf)(char uplo, int n, ntype *a, int lda, int *info);
/* Matrix inverse based on Cholesky factorization */
void THZLapack_(potri)(char uplo, int n, ntype *a, int lda, int *info);
/* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
void THZLapack_(potrs)(char uplo, int n, int nrhs, ntype *a, int lda, ntype *b, int ldb, int *info);
/* Cholesky factorization with complete pivoting. */
void THZLapack_(pstrf)(char uplo, int n, ntype *a, int lda, int *piv, int *rank, ntype tol, ntype *work, int *info);

/* QR decomposition */
void THZLapack_(geqrf)(int m, int n, ntype *a, int lda, ntype *tau, ntype *work, int lwork, int *info);
/* Build Q from output of geqrf */
void THZLapack_(orgqr)(int m, int n, int k, ntype *a, int lda, ntype *tau, ntype *work, int lwork, int *info);
/* Multiply Q with a matrix from output of geqrf */
void THZLapack_(ormqr)(char side, char trans, int m, int n, int k, ntype *a, int lda, ntype *tau, ntype *c, int ldc, ntype *work, int lwork, int *info);

#endif
