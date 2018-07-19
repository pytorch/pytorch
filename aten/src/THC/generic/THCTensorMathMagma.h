#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathMagma.h"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

// MAGMA (i.e. CUDA implementation of LAPACK functions)
THC_API void THCTensor_(gesv)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_);
THC_API void THCTensor_(trtrs)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_,
                               const char *uplo, const char *trans, const char *diag);
THC_API void THCTensor_(gels)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_);
THC_API void THCTensor_(syev)(THCState *state, THCTensor *re_, THCTensor *rv_, THCTensor *a_, const char *jobz, const char *uplo);
THC_API void THCTensor_(geev)(THCState *state, THCTensor *re_, THCTensor *rv_, THCTensor *a_, const char *jobvr);
THC_API void THCTensor_(gesvd)(THCState *state, THCTensor *ru_, THCTensor *rs_, THCTensor *rv_, THCTensor *a, const char *jobu);
THC_API void THCTensor_(gesvd2)(THCState *state, THCTensor *ru_, THCTensor *rs_, THCTensor *rv_, THCTensor *ra_, THCTensor *a, const char *jobu);
THC_API void THCTensor_(getri)(THCState *state, THCTensor *ra_, THCTensor *a);
THC_API void THCTensor_(potri)(THCState *state, THCTensor *ra_, THCTensor *a, const char *uplo);
THC_API void THCTensor_(potrf)(THCState *state, THCTensor *ra_, THCTensor *a, const char *uplo);
THC_API void THCTensor_(potrs)(THCState *state, THCTensor *rb_, THCTensor *a, THCTensor *b, const char *uplo);
THC_API void THCTensor_(geqrf)(THCState *state, THCTensor *ra_, THCTensor *rtau_, THCTensor *a_);
THC_API void THCTensor_(qr)(THCState *state, THCTensor *rq_, THCTensor *rr_, THCTensor *a);

#endif // defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

#endif
