#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensor.hpp"
#else

// This allows one to write generic functions (i.e. functions that work across all THRealTensor types)
// without having to explicitly cast the THRealTensor.
typedef struct THCTensor : _THCTensor {
} THCTensor;

#endif
