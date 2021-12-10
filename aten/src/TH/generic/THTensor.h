#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensor.h"
#else

/* a la lua? dim, storageoffset, ...  et les methodes ? */

#include <c10/core/TensorImpl.h>

#define THTensor at::TensorImpl

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THTensor type.
#define THFloatTensor THTensor
#define THDoubleTensor THTensor
#define THHalfTensor THTensor
#define THByteTensor THTensor
#define THCharTensor THTensor
#define THShortTensor THTensor
#define THIntTensor THTensor
#define THLongTensor THTensor
#define THBoolTensor THTensor
#define THBFloat16Tensor THTensor
#define THComplexFloatTensor THTensor
#define THComplexDoubleTensor THTensor

/**** creation methods ****/
TH_API THTensor *THTensor_(newWithStorage1d)(c10::StorageImpl *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_);

#endif
