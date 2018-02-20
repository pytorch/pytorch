#ifndef TH_VECTOR_INC
#define TH_VECTOR_INC

#include "THGeneral.h"
#include "THMath.h"

#define THVector_(NAME) TH_CONCAT_4(TH,Real,Vector_,NAME)

#ifdef __cplusplus
extern "C" {
#endif
/* We are going to use dynamic dispatch, and want only to generate declarations
 * of the vector functions */
#include "generic/THVector.h"
#include "THGenerateAllTypes.h"
#ifdef __cplusplus
}
#endif
#endif // TH_VECTOR_INC
