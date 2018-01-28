#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorLapack.h"
#else

TH_API void THZTensor_(gesv)(THZTensor *rb_, THZTensor *ra_, THZTensor *b_, THZTensor *a_);
TH_API void THZTensor_(trtrs)(THZTensor *rb_, THZTensor *ra_, THZTensor *b_, THZTensor *a_, const char *uplo, const char *trans, const char *diag);
TH_API void THZTensor_(gels)(THZTensor *rb_, THZTensor *ra_, THZTensor *b_, THZTensor *a_);
TH_API void THZTensor_(syev)(THZTensor *re_, THZTensor *rv_, THZTensor *a_, const char *jobz, const char *uplo);
TH_API void THZTensor_(geev)(THZTensor *re_, THZTensor *rv_, THZTensor *a_, const char *jobvr);
TH_API void THZTensor_(gesvd)(THZTensor *ru_, THZTensor *rs_, THZTensor *rv_, THZTensor *a, const char *jobu);
TH_API void THZTensor_(gesvd2)(THZTensor *ru_, THZTensor *rs_, THZTensor *rv_, THZTensor *ra_, THZTensor *a, const char *jobu);
TH_API void THZTensor_(getri)(THZTensor *ra_, THZTensor *a);
TH_API void THZTensor_(potrf)(THZTensor *ra_, THZTensor *a, const char *uplo);
TH_API void THZTensor_(potrs)(THZTensor *rb_, THZTensor *b_, THZTensor *a_,  const char *uplo);
TH_API void THZTensor_(potri)(THZTensor *ra_, THZTensor *a, const char *uplo);
TH_API void THZTensor_(qr)(THZTensor *rq_, THZTensor *rr_, THZTensor *a);
TH_API void THZTensor_(geqrf)(THZTensor *ra_, THZTensor *rtau_, THZTensor *a);
TH_API void THZTensor_(orgqr)(THZTensor *ra_, THZTensor *a, THZTensor *tau);
TH_API void THZTensor_(ormqr)(THZTensor *ra_, THZTensor *a, THZTensor *tau, THZTensor *c, const char *side, const char *trans);
TH_API void THZTensor_(pstrf)(THZTensor *ra_, THIntTensor *rpiv_, THZTensor*a, const char* uplo, ntype tol);

TH_API void THZTensor_(btrifact)(THZTensor *ra_, THIntTensor *rpivots_, THIntTensor *rinfo_, int pivot, THZTensor *a);
TH_API void THZTensor_(btrisolve)(THZTensor *rb_, THZTensor *b, THZTensor *atf, THIntTensor *pivots);

#endif
