#ifndef THP_SERIALIZATION_INC
#define THP_SERIALIZATION_INC

template <class io>
void doRead(io fildes, void* buf, size_t nbytes);

template <class io>
void doWrite(io fildes, void* buf, size_t nbytes);

// Note that this takes a mutable storage because it may ultimately be
// calling into a Python API which requires a char* parameter, even
// though it is only reading from it.
template <class io>
void THPStorage_writeFileRaw(
    c10::StorageImpl* self,
    io fd,
    bool save_size,
    uint64_t element_size);

template <class io>
c10::intrusive_ptr<c10::StorageImpl> THPStorage_readFileRaw(
    io fd,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size);

#endif
