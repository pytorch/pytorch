#ifndef THP_CUDNN_HANDLE_INC
#define THP_CUDNN_HANDLE_INC

#include <cudnn.h>
#include "THC/THC.h"
#include "Types.h"

namespace torch { namespace cudnn {

cudnnHandle_t getCudnnHandle();
void useCurrentStream(cudnnHandle_t handle, THCState *state);

}} // namespace

#endif
