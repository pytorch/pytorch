#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLapack.h"
#else



/* AX=B */
void THLapack_(gesv)(int n, int nrhs, real *a, int lda, int *ipiv, real *b, int ldb, int* info);
/* ||AX-B|| */
void THLapack_(gels)(char trans, int m, int n, int nrhs, real *a, int lda, real *b, int ldb, real *work, int lwork, int *info);
/* Eigenvals */
void THLapack_(syev)(char jobz, char uplo, int n, real *a, int lda, real *w, real *work, int lwork, int *info);
/* Non-sym eigenvals */
void THLapack_(geev)(char jobvl, char jobvr, int n, real *a, int lda, real *wr, real *wi, real* vl, int ldvl, real *vr, int ldvr, real *work, int lwork, int *info);
/* svd */
void THLapack_(gesvd)(char jobu, char jobvt, int m, int n, real *a, int lda, real *s, real *u, int ldu, real *vt, int ldvt, real *work, int lwork, int *info);
/* LU decomposition */
void THLapack_(getrf)(int m, int n, real *a, int lda, int *ipiv, int *info);
/* Matrix Inverse */
void THLapack_(getri)(int n, real *a, int lda, int *ipiv, real *work, int lwork, int* info);
#endif
