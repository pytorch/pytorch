#include <THC/THCGeneral.h>
#include <THC/THCTensor.hpp>

#include <new>

#include <THC/generic/THCTensor.cpp>
#include <THC/THCGenerateByteType.h>

void THCTensor_setStorage(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_)
{
  c10::raw::intrusive_ptr::incref(storage_);
  THTensor_wrap(self).set_(at::Storage(c10::intrusive_ptr<at::StorageImpl>::reclaim(storage_)),
                           storageOffset_, size_, stride_);
}
