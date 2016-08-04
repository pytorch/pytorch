#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMathDispatch.h"
#else

// NOTE This header will contain the declarations for the dispatch stubs the user actually calls,
// which will be defined in generic/THCpuDispatchInit.c

TH_API int THTensor_(cpuDispatchInit)();

TH_API void THTensor_(add)(THTensor *r_, THTensor *t, real value);

#endif
