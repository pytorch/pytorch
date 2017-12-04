#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.h"
#else

void THPStorage_(writeFileRaw)(THStorage *self, int fd);
THStorage * THPStorage_(readFileRaw)(int fd, THStorage *storage);

#endif
