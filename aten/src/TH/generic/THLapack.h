#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THLapack.h"
#else

/* ||AX-B|| */
TH_API void THLapack_(gels)(char trans, int m, int n, int nrhs, scalar_t *a, int lda, scalar_t *b, int ldb, scalar_t *work, int lwork, int *info);

#endif
