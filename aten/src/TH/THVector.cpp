#include "THVector.h"

#include "vector/simd.h"

#ifdef __NEON__
#include "vector/NEON.cpp"
#endif

#ifdef __PPC64__
#include "vector/VSX.cpp"
#endif

#if defined(USE_AVX)
#include "vector/AVX.h"
#endif

#if defined(USE_AVX2)
#include "vector/AVX2.h"
#endif

#include "generic/THVectorDefault.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THVectorDispatch.cpp"
#include "THGenerateAllTypes.h"
