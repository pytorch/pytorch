#include <Python.h>
#include "THDP.h"

#include "override_macros.h"

template<>
void THPPointer<THDTensorDescriptor>::free() {
  if (ptr)
    THDTensorDescriptor_free(ptr);
}
template class THPPointer<THDTensorDescriptor>;

#define THD_GENERIC_FILE "torch/csrc/generic/utils.cpp"
#include <THD/base/THDGenerateAllTypes.h>
