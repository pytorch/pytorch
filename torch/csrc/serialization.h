#ifndef THP_SERIALIZATION_INC
#define THP_SERIALIZATION_INC

#include "generic/serialization.h"
#include <TH/THGenerateAllTypes.h>

#include "generic/serialization.h"
#include <TH/THGenerateHalfType.h>

template <class io>
ssize_t doRead(io fildes, void* buf, size_t nbytes);

template <class io>
ssize_t doWrite(io fildes, void* buf, size_t nbytes);

#endif
