#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "storages/generic/THCStorage.hpp"
#else

template<>
struct thc_storage_traits<real> {
  using storage_type = THCRealStorage;
};

#endif
