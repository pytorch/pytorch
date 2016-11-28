#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "base/storages/generic/THStorage.hpp"
#else

template<>
struct th_storage_traits<real> {
  using storage_type = THRealStorage;
};

#endif
