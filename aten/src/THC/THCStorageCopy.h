#ifndef THC_STORAGE_COPY_INC
#define THC_STORAGE_COPY_INC

#include <THC/THCStorage.h>
#include <THC/THCGeneral.h>
#include <TH/THHalf.h>
#include <ATen/core/Tensor.h>

#include <THC/generic/THCStorageCopy.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCStorageCopy.h>
#include <THC/THCGenerateComplexTypes.h>

#include <THC/generic/THCStorageCopy.h>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCStorageCopy.h>
#include <THC/THCGenerateBFloat16Type.h>

// reclaim tensor ownership from an owning raw pointer
inline at::Tensor tensor_reclaim(c10::TensorImpl *tensor) {
  return at::Tensor(c10::intrusive_ptr<c10::TensorImpl>::reclaim(tensor));
}

#endif
