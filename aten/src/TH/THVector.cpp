#include <TH/THVector.h>

#include <TH/vector/simd.h>

#ifdef __NEON__
#include <TH/vector/NEON.cpp>
#endif

#ifdef __PPC64__
#include <TH/vector/VSX.cpp>
#endif

#if defined(USE_AVX)
#include <TH/vector/AVX.h>
#endif

#if defined(USE_AVX2)
#include <TH/vector/AVX2.h>
#endif

#include <TH/generic/THVectorDefault.cpp>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THVectorDefault.cpp>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THVectorDefault.cpp>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THVectorDefault.cpp>
#include <TH/THGenerateBFloat16Type.h>

#include <TH/generic/THVectorDispatch.cpp>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THVectorDispatch.cpp>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THVectorDispatch.cpp>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THVectorDispatch.cpp>
#include <TH/THGenerateBFloat16Type.h>
