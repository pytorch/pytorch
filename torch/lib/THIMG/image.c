#include <TH.h>
#include "THIMG.h"

#ifdef max
#undef max
#endif
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )

#ifdef min
#undef min
#endif
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

#include "font.c"

#define THIMG_ARGCHECK(COND, ARG, T, FORMAT)  \
  if (!(COND)) {                              \
    THDescBuff s1 = THTensor_(sizeDesc)(T);   \
    THArgCheck(COND, ARG, FORMAT, s1.str);    \
  }

#include "generic/image.c"
#include "THGenerateAllTypes.h"
