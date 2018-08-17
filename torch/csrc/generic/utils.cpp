#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.cpp"
#else

#if defined(THD_GENERIC_FILE) || defined(TH_REAL_IS_HALF)
#define GENERATE_SPARSE 0
#else
#define GENERATE_SPARSE 1
#endif

template<>
void THPPointer<THPStorage>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<THPStorage>;

#undef GENERATE_SPARSE

#endif
