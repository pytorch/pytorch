#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/serialization.h"
#else

template <class io>
void THPStorage_(writeFileRaw)(c10::StorageImpl *self, io fd, bool save_size, uint64_t element_size);

template <class io>
c10::intrusive_ptr<c10::StorageImpl> THPStorage_(readFileRaw)(
    io fd, c10::intrusive_ptr<c10::StorageImpl> storage, uint64_t element_size);

#endif
