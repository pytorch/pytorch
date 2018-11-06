#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.cpp"
#else

#include "ATen/native/Copy.h"

void THTensor_(copyTranspose)(THTensor *tensor, THTensor *src) {
  at::Tensor tensor_wrap = THTensor_wrap(tensor);
  at::Tensor src_wrap = THTensor_wrap(src);
  at::native::_copy_same_type_transpose(tensor_wrap, src_wrap);
}

void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  at::Tensor tensor_wrap = THTensor_wrap(tensor);
  at::Tensor src_wrap = THTensor_wrap(src);
  at::native::_copy_same_type(tensor_wrap, src_wrap);
}

#endif
