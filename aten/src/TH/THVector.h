#ifndef TH_VECTOR_INC
#define TH_VECTOR_INC

#include <TH/THGeneral.h>
#include <TH/THMath.h>

#define THVector_(NAME) TH_CONCAT_4(TH,Real,Vector_,NAME)

/* We are going to use dynamic dispatch, and want only to generate declarations
 * of the vector functions */
#include <TH/generic/THVector.h>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THVector.h>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THVector.h>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THVector.h>
#include <TH/THGenerateBFloat16Type.h>

#endif // TH_VECTOR_INC
