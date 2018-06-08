#include <cmath>
#include <float.h>

#include <atomic>
#include "THTensor.hpp"
#include "THVector.h"
#include "generic/simd/simd.h"

#include "THBlas.h"
#include "THLapack.h"
#include "THRandom.h"
#include "THTensorDimApply.h"
#include "THMath.h"

#include "generic/THTensor.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensor.cpp"
#include "THGenerateHalfType.h"

#include "generic/THTensorCopy.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensorCopy.cpp"
#include "THGenerateHalfType.h"

#include "generic/THTensorRandom.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensorMath.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensorConv.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensorLapack.cpp"
#include "THGenerateFloatTypes.h"

void THTensor_free(THTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(--self->refcount == 0)
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THStorage_free(self->storage);
      self->refcount.~atomic<int>();
      THFree(self);
    }
  }
}
