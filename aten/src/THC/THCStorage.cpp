#include "THCStorage.hpp"
#include "THCGeneral.h"

#include "THCHalf.h"

#include <new>

#include "generic/THCStorage.cpp"
#include "THCGenerateAllTypes.h"

void THCStorage_free(THCState *state, THCStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (--self->refcount == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THCudaCheck(
        (*self->allocator->free)(self->allocatorContext, self->data_ptr));
    }
    if(self->flag & TH_STORAGE_VIEW) {
      THCStorage_free(state, self->view);
    }
    self->refcount.~atomic<int>();
    THFree(self);
  }
}
