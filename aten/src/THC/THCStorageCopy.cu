#include <THC/THCGeneral.h>
#include <THC/THCStorageCopy.h>

#include <TH/THHalf.h>
#include <THC/THCStorage.hpp>
#include <THC/THCTensor.hpp>
#include <THC/THCTensorCopy.h>

// clang-format off
#include <THC/generic/THCStorageCopy.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCStorageCopy.cu>
#include <THC/THCGenerateComplexTypes.h>

#include <THC/generic/THCStorageCopy.cu>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCStorageCopy.cu>
#include <THC/THCGenerateBFloat16Type.h>
// clang-format on
