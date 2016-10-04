#include "TH.h"
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

#include "generic/image.c"
#include "THGenerateAllTypes.h"
