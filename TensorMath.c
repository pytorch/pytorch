#include "TH.h"
#include "luaT.h"
#include "utils.h"

#include "sys/time.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)

static const void* torch_ByteTensor_id;
static const void* torch_CharTensor_id;
static const void* torch_ShortTensor_id;
static const void* torch_IntTensor_id;
static const void* torch_LongTensor_id;
static const void* torch_FloatTensor_id;
static const void* torch_DoubleTensor_id;

static const void* torch_LongStorage_id;


#include "TensorMathWrap.c"
//#include "TensorLapackWrap.c"
//#include "TensorConvWrap.c"

//#include "generic/TensorLapack.c"
//#include "THGenerateFloatTypes.h"

//#include "generic/TensorConv.c"
//#include "THGenerateAllTypes.h"

void torch_TensorMath_init(lua_State *L)
{
  torch_ByteTensorMath_init(L);
  torch_CharTensorMath_init(L);
  torch_ShortTensorMath_init(L);
  torch_IntTensorMath_init(L);
  torch_LongTensorMath_init(L);
  torch_FloatTensorMath_init(L);
  torch_DoubleTensorMath_init(L);
  luaL_register(L, NULL, torch_TensorMath__);

/*   torch_FloatLapack_init(L); */
/*   torch_DoubleLapack_init(L); */
/*   luaL_register(L, NULL, torch_TensorLapack__); */

/*   torch_ByteConv_init(L); */
/*   torch_CharConv_init(L); */
/*   torch_ShortConv_init(L); */
/*   torch_IntConv_init(L); */
/*   torch_LongConv_init(L); */
/*   torch_FloatConv_init(L); */
/*   torch_DoubleConv_init(L); */
/*   luaL_register(L, NULL, torch_TensorConv__); */
}
