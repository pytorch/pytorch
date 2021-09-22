#include <THC/THCStorageCopy.h>
#include <THC/THCGeneral.h>

#include <TH/THHalf.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensor.hpp>
#include <THC/THCStorage.hpp>

// reclaim tensor ownership from an owning raw pointer
static inline at::Tensor tensor_reclaim(c10::TensorImpl *tensor) {
  return at::Tensor(c10::intrusive_ptr<c10::TensorImpl>::reclaim(tensor));
}

#include <THC/generic/THCStorageCopy.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCStorageCopy.cu>
#include <THC/THCGenerateComplexTypes.h>

#include <THC/generic/THCStorageCopy.cu>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCStorageCopy.cu>
#include <THC/THCGenerateBFloat16Type.h>
