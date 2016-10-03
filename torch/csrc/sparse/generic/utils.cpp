#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "torch/csrc/sparse/generic/utils.cpp"
#else

template<>
void THPPointer<THSTensor>::free() {
  if (ptr)
    THSTensor_(free)(LIBRARY_STATE ptr);
}

template<>
void THPPointer<THSPTensor>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<THTensor>;
template class THPPointer<THPTensor>;

#endif
