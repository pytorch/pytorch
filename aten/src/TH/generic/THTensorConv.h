#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorConv.h"
#else

TH_API void THTensor_(validXCorr2Dptr)(scalar_t *r_,
                                    scalar_t alpha,
                                    scalar_t *t_, int64_t ir, int64_t ic,
                                    scalar_t *k_, int64_t kr, int64_t kc,
                                    int64_t sr, int64_t sc);

TH_API void THTensor_(validConv2Dptr)(scalar_t *r_,
                                   scalar_t alpha,
                                   scalar_t *t_, int64_t ir, int64_t ic,
                                   scalar_t *k_, int64_t kr, int64_t kc,
                                   int64_t sr, int64_t sc);

TH_API void THTensor_(fullXCorr2Dptr)(scalar_t *r_,
                                   scalar_t alpha,
                                   scalar_t *t_, int64_t ir, int64_t ic,
                                   scalar_t *k_, int64_t kr, int64_t kc,
                                   int64_t sr, int64_t sc);

TH_API void THTensor_(fullConv2Dptr)(scalar_t *r_,
                                  scalar_t alpha,
                                  scalar_t *t_, int64_t ir, int64_t ic,
                                  scalar_t *k_, int64_t kr, int64_t kc,
                                  int64_t sr, int64_t sc);

TH_API void THTensor_(validXCorr2DRevptr)(scalar_t *r_,
                                       scalar_t alpha,
                                       scalar_t *t_, int64_t ir, int64_t ic,
                                       scalar_t *k_, int64_t kr, int64_t kc,
                                       int64_t sr, int64_t sc);

TH_API void THTensor_(conv2DRevger)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol);
TH_API void THTensor_(conv2DRevgerm)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol);
TH_API void THTensor_(conv2Dger)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv2Dmv)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv2Dmm)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv2Dmul)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv2Dcmul)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc);

TH_API void THTensor_(validXCorr3Dptr)(scalar_t *r_,
                                    scalar_t alpha,
                                    scalar_t *t_, int64_t it, int64_t ir, int64_t ic,
                                    scalar_t *k_, int64_t kt, int64_t kr, int64_t kc,
                                    int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(validConv3Dptr)(scalar_t *r_,
                                   scalar_t alpha,
                                   scalar_t *t_, int64_t it, int64_t ir, int64_t ic,
                                   scalar_t *k_, int64_t kt, int64_t kr, int64_t kc,
                                   int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(fullXCorr3Dptr)(scalar_t *r_,
                                   scalar_t alpha,
                                   scalar_t *t_, int64_t it, int64_t ir, int64_t ic,
                                   scalar_t *k_, int64_t kt, int64_t kr, int64_t kc,
                                   int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(fullConv3Dptr)(scalar_t *r_,
                                  scalar_t alpha,
                                  scalar_t *t_, int64_t it, int64_t ir, int64_t ic,
                                  scalar_t *k_, int64_t kt, int64_t kr, int64_t kc,
                                  int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(validXCorr3DRevptr)(scalar_t *r_,
                                       scalar_t alpha,
                                       scalar_t *t_, int64_t it, int64_t ir, int64_t ic,
                                       scalar_t *k_, int64_t kt, int64_t kr, int64_t kc,
                                       int64_t st, int64_t sr, int64_t sc);

TH_API void THTensor_(conv3DRevger)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol);
TH_API void THTensor_(conv3Dger)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv3Dmv)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv3Dmul)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);
TH_API void THTensor_(conv3Dcmul)(THTensor *r_, scalar_t beta, scalar_t alpha, THTensor *t_, THTensor *k_, int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc);

#endif
