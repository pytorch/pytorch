#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorLapack.h"
#else

THD_API void THDTensor_(gesv)(THDTensor *rb_, THDTensor *ra_, THDTensor *b_, THDTensor *a_);
THD_API void THDTensor_(trtrs)(THDTensor *rb_, THDTensor *ra_, THDTensor *b_, THDTensor *a_,
                               const char *uplo, const char *trans, const char *diag);
THD_API void THDTensor_(gels)(THDTensor *rb_, THDTensor *ra_, THDTensor *b_, THDTensor *a_);
THD_API void THDTensor_(syev)(THDTensor *re_, THDTensor *rv_, THDTensor *a_,
                              const char *jobz, const char *uplo);
THD_API void THDTensor_(geev)(THDTensor *re_, THDTensor *rv_, THDTensor *a_, const char *jobvr);
THD_API void THDTensor_(gesvd)(THDTensor *ru_, THDTensor *rs_, THDTensor *rv_, THDTensor *a,
                               const char *jobu);
THD_API void THDTensor_(gesvd2)(THDTensor *ru_, THDTensor *rs_, THDTensor *rv_, THDTensor *ra_,
                                THDTensor *a, const char *jobu);
THD_API void THDTensor_(getri)(THDTensor *ra_, THDTensor *a);
THD_API void THDTensor_(potrf)(THDTensor *ra_, THDTensor *a, const char *uplo);
THD_API void THDTensor_(potrs)(THDTensor *rb_, THDTensor *b_, THDTensor *a_,  const char *uplo);
THD_API void THDTensor_(potri)(THDTensor *ra_, THDTensor *a, const char *uplo);
THD_API void THDTensor_(qr)(THDTensor *rq_, THDTensor *rr_, THDTensor *a);
THD_API void THDTensor_(geqrf)(THDTensor *ra_, THDTensor *rtau_, THDTensor *a);
THD_API void THDTensor_(orgqr)(THDTensor *ra_, THDTensor *a, THDTensor *tau);
THD_API void THDTensor_(ormqr)(THDTensor *ra_, THDTensor *a, THDTensor *tau, THDTensor *c,
                               const char *side, const char *trans);
THD_API void THDTensor_(pstrf)(THDTensor *ra_, THDIntTensor *rpiv_, THDTensor*a,
                               const char* uplo, real tol);
#endif
