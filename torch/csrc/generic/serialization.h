#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.h"
#else

template <class io>
void THPStorage_(writeFileRaw)(THWStorageImpl *self, io fd);

template <class io>
THWStorageImpl * THPStorage_(readFileRaw)(io fd, THWStorageImpl *storage);

#endif
