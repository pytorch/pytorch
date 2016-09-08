#ifndef TH_VECTOR_INC
#define TH_VECTOR_INC

#include "THGeneral.h"

#define THVector_(NAME) TH_CONCAT_4(TH,Real,Vector_,NAME)

/* We are going to use dynamic dispatch, and want only to generate declarations
 * of the vector functions */
#include "generic/THVector.h"
#include "THGenerateAllTypes.h"

#endif // TH_VECTOR_INC
