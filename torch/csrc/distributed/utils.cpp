#include "utils.h"

template<>
void THPPointer<THDTensorDescriptor>::free() {
  if (ptr)
    THDTensorDescriptor_free(ptr);
}

template class THPPointer<THDTensorDescriptor>;
