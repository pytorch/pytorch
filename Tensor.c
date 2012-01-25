#include "general.h"

static const void *torch_File_id = NULL;

static const void *torch_ByteStorage_id = NULL;
static const void *torch_CharStorage_id = NULL;
static const void *torch_ShortStorage_id = NULL;
static const void *torch_IntStorage_id = NULL;
static const void *torch_LongStorage_id = NULL;
static const void *torch_FloatStorage_id = NULL;
static const void *torch_DoubleStorage_id = NULL;

static const void *torch_ByteTensor_id = NULL;
static const void *torch_CharTensor_id = NULL;
static const void *torch_ShortTensor_id = NULL;
static const void *torch_IntTensor_id = NULL;
static const void *torch_LongTensor_id = NULL;
static const void *torch_FloatTensor_id = NULL;
static const void *torch_DoubleTensor_id = NULL;

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
#define torch_Storage_id TH_CONCAT_3(torch_,Real,Storage_id)
#define STRING_torchStorage TH_CONCAT_STRING_3(torch.,Real,Storage)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_,Real,Tensor_,NAME)
#define torch_Tensor_id TH_CONCAT_3(torch_,Real,Tensor_id)
#define STRING_torchTensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

#include "generic/Tensor.c"
#include "THGenerateAllTypes.h"
