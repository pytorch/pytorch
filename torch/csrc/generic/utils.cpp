#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.cpp"
#else

#if defined(THD_GENERIC_FILE) || defined(TH_REAL_IS_HALF)
#define GENERATE_SPARSE 0
#else
#define GENERATE_SPARSE 1
#endif

template<>
void THPPointer<THStorage>::free() {
  if (ptr)
    THStorage_(free)(LIBRARY_STATE ptr);
}

template<>
void THPPointer<THTensor>::free() {
  if (ptr)
    THTensor_(free)(LIBRARY_STATE ptr);
}

template<>
void THPPointer<THPStorage>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template<>
void THPPointer<THPTensor>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

#if GENERATE_SPARSE
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
#endif


template class THPPointer<THStorage>;
template class THPPointer<THTensor>;
template class THPPointer<THPStorage>;
template class THPPointer<THPTensor>;
#if GENERATE_SPARSE
template class THPPointer<THSTensor>;
template class THPPointer<THSPTensor>;
#endif

#undef GENERATE_SPARSE

#endif
