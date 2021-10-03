#include <TH/THTensor.hpp>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THTensor.cpp>
#include <TH/THGenerateAllTypes.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THTensor.cpp>
#include <TH/THGenerateComplexTypes.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THTensor.cpp>
#include <TH/THGenerateHalfType.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THTensor.cpp>
#include <TH/THGenerateBoolType.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THTensor.cpp>
#include <TH/THGenerateBFloat16Type.h>

#include <ATen/native/Resize.h>
#include <ATen/TensorUtils.h>

#include <numeric>

// NB: This is NOT valid on UndefinedTensorImpl
void THTensor_free(THTensor *self)
{
  if (!self) return;
  c10::raw::intrusive_ptr::decref(self);
}

void THTensor_setStorage(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_) {
  c10::raw::intrusive_ptr::incref(storage_);
  THTensor_wrap(self).set_(at::Storage(c10::intrusive_ptr<at::StorageImpl>::reclaim(storage_)), storageOffset_, size_, stride_);
}
