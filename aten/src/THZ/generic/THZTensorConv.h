#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorConv.h"
#else

TH_API void THZTensor_(validXCorr2Dptr)(ntype *r_,
                                    ntype alpha,
                                    ntype *t_, int64_t ir, int64_t ic,
                                    ntype *k_, int64_t kr, int64_t kc,
                                    int64_t sr, int64_t sc);

TH_API void THZTensor_(validConv2Dptr)(ntype *r_,
                                   ntype alpha,
                                   ntype *t_, int64_t ir, int64_t ic,
                                   ntype *k_, int64_t kr, int64_t kc,
                                   int64_t sr, int64_t sc);

TH_API void THZTensor_(fullXCorr2Dptr)(ntype *r_,
                                   ntype alpha,
                                   ntype *t_, int64_t ir, int64_t ic,
                                   ntype *k_, int64_t kr, int64_t kc,
                                   int64_t sr, int64_t sc);

TH_API void THZTensor_(fullConv2Dptr)(ntype *r_,
                                  ntype alpha,
                                  ntype *t_, int64_t ir, int64_t ic,
                                  ntype *k_, int64_t kr, int64_t kc,
                                  int64_t sr, int64_t sc);

TH_API void THZTensor_(validXCorr2DRevptr)(ntype *r_,
                                       ntype alpha,
                                       ntype *t_, int64_t ir, int64_t ic,
                                       ntype *k_, int64_t kr, int64_t kc,
                                       int64_t sr, int64_t sc);

TH_API void THZTensor_(conv2DRevger)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t srow, int64_t scol);
TH_API void THZTensor_(conv2DRevgerm)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t srow, int64_t scol);
TH_API void THZTensor_(conv2Dger)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THZTensor_(conv2Dmv)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THZTensor_(conv2Dmm)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THZTensor_(conv2Dmul)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THZTensor_(conv2Dcmul)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);

TH_API void THZTensor_(validXCorr3Dptr)(ntype *r_,
                                    ntype alpha,
                                    ntype *t_, int64_t it, int64_t ir, int64_t ic,
                                    ntype *k_, int64_t kt, int64_t kr, int64_t kc,
                                    int64_t st, int64_t sr, int64_t sc);

TH_API void THZTensor_(validConv3Dptr)(ntype *r_,
                                   ntype alpha,
                                   ntype *t_, int64_t it, int64_t ir, int64_t ic,
                                   ntype *k_, int64_t kt, int64_t kr, int64_t kc,
                                   int64_t st, int64_t sr, int64_t sc);

TH_API void THZTensor_(fullXCorr3Dptr)(ntype *r_,
                                   ntype alpha,
                                   ntype *t_, int64_t it, int64_t ir, int64_t ic,
                                   ntype *k_, int64_t kt, int64_t kr, int64_t kc,
                                   int64_t st, int64_t sr, int64_t sc);

TH_API void THZTensor_(fullConv3Dptr)(ntype *r_,
                                  ntype alpha,
                                  ntype *t_, int64_t it, int64_t ir, int64_t ic,
                                  ntype *k_, int64_t kt, int64_t kr, int64_t kc,
                                  int64_t st, int64_t sr, int64_t sc);

TH_API void THZTensor_(validXCorr3DRevptr)(ntype *r_,
                                       ntype alpha,
                                       ntype *t_, int64_t it, int64_t ir, int64_t ic,
                                       ntype *k_, int64_t kt, int64_t kr, int64_t kc,
                                       int64_t st, int64_t sr, int64_t sc);

TH_API void THZTensor_(conv3DRevger)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t sdepth, int64_t srow, int64_t scol);
TH_API void THZTensor_(conv3Dger)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THZTensor_(conv3Dmv)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THZTensor_(conv3Dmul)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THZTensor_(conv3Dcmul)(THZTensor *r_, ntype beta, ntype alpha, THZTensor *t_, THZTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);

#endif
