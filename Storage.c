#include "general.h"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
#define THFile_readRealRaw TH_CONCAT_3(THFile_read, Real, Raw)
#define THFile_writeRealRaw TH_CONCAT_3(THFile_write, Real, Raw)
#define torch_Storage TH_CONCAT_STRING_3(torch.,Real,Storage)

#include "generic/Storage.c"
#include "THGenerateAllTypes.h"
