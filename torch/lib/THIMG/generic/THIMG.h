#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THIMG.h"
#else

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
          long x1long,
          long y1long,
          long x2long,
          long y2long,
          int lineWidth,
          int cr,
          int cg,
          int cb);

#endif
