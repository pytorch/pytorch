#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THIMG.h"
#else

TH_API void THIMG_(Main_scaleBilinear)(
          THTensor *Tsrc,
          THTensor *Tdst);
TH_API void THIMG_(Main_scaleBicubic)(
          THTensor *Tsrc,
          THTensor *Tdst);
TH_API void THIMG_(Main_scaleSimple)(
          THTensor *Tsrc,
          THTensor *Tdst);

TH_API void THIMG_(Main_rotate)(
          THTensor *Tsrc,
          THTensor *Tdst,
          float theta);
TH_API void THIMG_(Main_rotateBilinear)(
          THTensor *Tsrc,
          THTensor *Tdst,
          float theta);

TH_API void THIMG_(Main_polar)(
          THTensor *Tsrc,
          THTensor *Tdst,
          bool doFull);
TH_API void THIMG_(Main_polarBilinear)(
          THTensor *Tsrc,
          THTensor *Tdst,
          bool doFull);

TH_API void THIMG_(Main_logPolar)(
          THTensor *Tsrc,
          THTensor *Tdst,
          bool doFull);
TH_API void THIMG_(Main_logPolarBilinear)(
          THTensor *Tsrc,
          THTensor *Tdst,
          bool doFull);

TH_API void THIMG_(Main_cropNoScale)(
          THTensor *Tsrc,
          THTensor *Tdst,
          long startx,
          long starty);

TH_API void THIMG_(Main_translate)(
          THTensor *Tsrc,
          THTensor *Tdst,
          long shiftx,
          long shifty);

TH_API void THIMG_(Main_saturate)(
          THTensor *input);

TH_API void THIMG_(Main_rgb2hsl)(
          THTensor *rgb,
          THTensor *hsl);
TH_API void THIMG_(Main_hsl2rgb)(
          THTensor *hsl,
          THTensor *rgb);
TH_API void THIMG_(Main_rgb2hsv)(
          THTensor *rgb,
          THTensor *hsv);
TH_API void THIMG_(Main_hsv2rgb)(
          THTensor *hsv,
          THTensor *rgb);
TH_API void THIMG_(Main_rgb2lab)(
          THTensor *rgb,
          THTensor *lab);
TH_API void THIMG_(Main_lab2rgb)(
          THTensor *lab,
          THTensor *rgb);

TH_API void THIMG_(Main_vflip)(
          THTensor *dst,
          THTensor *src);
TH_API void THIMG_(Main_hflip)(
          THTensor *dst,
          THTensor *src);
TH_API void THIMG_(Main_flip)(
          THTensor *dst,
          THTensor *src,
          int flip_dim);

TH_API void THIMG_(Main_warp)(
          THTensor *dst,
          THTensor *src,
          THTensor *flowfield,
          int mode,
          int offset_mode,
          int clamp_mode,
          real pad_value);

TH_API void THIMG_(Main_gaussian)(
          THTensor *dst,
          float amplitude,
          bool normalize,
          float sigma_u,
          float sigma_v,
          float mean_u,
          float mean_v);;

TH_API void THIMG_(Main_colorize)(
          THTensor *output,
          THTensor *input,
          THTensor *colormap);

TH_API void THIMG_(Main_rgb2y)(
          THTensor *rgb,
          THTensor *yim);

TH_API void THIMG_(Main_drawtext)(
          THTensor *output,
          const char *text,
          long x,
          long y,
          int size,
          int cr,
          int cg,
          int cb,
          int bg_cr,
          int bg_cg,
          int bg_cb,
          bool wrap);

TH_API void THIMG_(Main_drawRect)(
          THTensor *output,
          int x1,
          int y1,
          int x2,
          int y2,
          int lineWidth,
          int cr,
          int cg,
          int cb);

#endif
