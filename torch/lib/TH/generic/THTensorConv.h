#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorConv.h"
#else

TH_API void THTensor_(validXCorr2Dptr)(real *r_,
                                    real alpha,
                                    real *t_, int64_t ir, int64_t ic,
                                    real *k_, int64_t kr, int64_t kc,
                                    int64_t sr, int64_t sc);

TH_API void THTensor_(validConv2Dptr)(real *r_,
                                   real alpha,
                                   real *t_, int64_t ir, int64_t ic,
                                   real *k_, int64_t kr, int64_t kc,
                                   int64_t sr, int64_t sc);

TH_API void THTensor_(fullXCorr2Dptr)(real *r_,
                                   real alpha,
                                   real *t_, int64_t ir, int64_t ic,
                                   real *k_, int64_t kr, int64_t kc,
                                   int64_t sr, int64_t sc);

TH_API void THTensor_(fullConv2Dptr)(real *r_,
                                  real alpha,
                                  real *t_, int64_t ir, int64_t ic,
                                  real *k_, int64_t kr, int64_t kc,
                                  int64_t sr, int64_t sc);

TH_API void THTensor_(validXCorr2DRevptr)(real *r_,
                                       real alpha,
                                       real *t_, int64_t ir, int64_t ic,
                                       real *k_, int64_t kr, int64_t kc,
                                       int64_t sr, int64_t sc);

TH_API void THTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol);
TH_API void THTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol);
TH_API void THTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv2Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv2Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);

TH_API void THTensor_(validXCorr3Dptr)(real *r_,
                                    real alpha,
                                    real *t_, int64_t it, int64_t ir, int64_t ic,
                                    real *k_, int64_t kt, int64_t kr, int64_t kc,
                                    int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(validConv3Dptr)(real *r_,
                                   real alpha,
                                   real *t_, int64_t it, int64_t ir, int64_t ic,
                                   real *k_, int64_t kt, int64_t kr, int64_t kc,
                                   int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(fullXCorr3Dptr)(real *r_,
                                   real alpha,
                                   real *t_, int64_t it, int64_t ir, int64_t ic,
                                   real *k_, int64_t kt, int64_t kr, int64_t kc,
                                   int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(fullConv3Dptr)(real *r_,
                                  real alpha,
                                  real *t_, int64_t it, int64_t ir, int64_t ic,
                                  real *k_, int64_t kt, int64_t kr, int64_t kc,
                                  int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(validXCorr3DRevptr)(real *r_,
                                       real alpha,
                                       real *t_, int64_t it, int64_t ir, int64_t ic,
                                       real *k_, int64_t kt, int64_t kr, int64_t kc,
                                       int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(conv3DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol);
TH_API void THTensor_(conv3Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv3Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv3Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv3Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);

#endif
