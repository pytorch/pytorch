#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.hpp"
#else

typedef struct THStorage
{
    at::ScalarType scalar_type;
    void *data_ptr;
    ptrdiff_t size;
    std::atomic<int> refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THStorage *view;

    template <typename T>
    inline T * data() const {
      auto scalar_type_T = at::CTypeToScalarType<th::from_type<T>>::to();
      if (scalar_type != scalar_type_T) {
        AT_ERROR("Attempt to access Storage having data type ", at::toString(scalar_type),
                 " as data type ", at::toString(scalar_type_T));
      }
      return unsafe_data<T>();
    }

    template <typename T>
    inline T * unsafe_data() const {
      return static_cast<T*>(this->data_ptr);
    }
} THStorage;

#endif
