#ifndef THP_SERIALIZATION_INC
#define THP_SERIALIZATION_INC

#include <torch/csrc/generic/serialization.h>
#include <TH/THGenerateAllTypes.h>

#include <torch/csrc/generic/serialization.h>
#include <TH/THGenerateHalfType.h>

#include <torch/csrc/generic/serialization.h>
#include <TH/THGenerateBoolType.h>

template <class io>
void doRead(io fildes, void* buf, size_t nbytes);

template <class io>
void doWrite(io fildes, void* buf, size_t nbytes);

#endif
