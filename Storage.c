#include "general.h"

static const void *torch_File_id = NULL;
static const void *torch_ByteStorage_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
#define torch_Storage_id TH_CONCAT_3(torch_,Real,Storage_id)
#define THFile_readRealRaw TH_CONCAT_3(THFile_read, Real, Raw)
#define THFile_writeRealRaw TH_CONCAT_3(THFile_write, Real, Raw)
#define STRING_torchStorage TH_CONCAT_STRING_3(torch.,Real,Storage)

#include "generic/Storage.c"
#include "THGenerateAllTypes.h"
