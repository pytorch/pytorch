#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.h"
#else

template <class io>
void THPStorage_(writeFileRaw)(THStorage *self, io fd);

template <class io>
THStorage * THPStorage_(readFileRaw)(io fd, THStorage *storage);

#endif
