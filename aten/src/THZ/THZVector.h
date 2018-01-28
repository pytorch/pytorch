#ifndef THZ_VECTOR_INC
#define THZ_VECTOR_INC

#include "THZGeneral.h"
#include "THZMath.h"

#define THZVector_(NAME) TH_CONCAT_4(TH,NType,Vector_,NAME)
#define THZPartVector_(NAME) TH_CONCAT_4(TH,Part,Vector_,NAME)

/* We are going to use dynamic dispatch, and want only to generate declarations
 * of the vector functions */
#include "generic/THZVector.h"
#include "THZGenerateAllTypes.h"

#endif // THZ_VECTOR_INC
