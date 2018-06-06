#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.hpp"
#else

// This allows one to write generic functions (i.e. functions that work across all THRealTensor types)
// without having to explicitly cast the THRealTensor.
typedef struct THTensor : _THTensor {
} THTensor;

#endif
