#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateQTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THQLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include <TH/THGenerateQUInt8Type.h>
#include <TH/THGenerateQInt8Type.h>
#include <TH/THGenerateQInt32Type.h>
#include <TH/THGenerateQUInt4x2Type.h>

#ifdef THQLocalGenerateManyTypes
#undef THQLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
