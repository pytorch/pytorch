#pragma once

#include "cudnn-wrapper.h"

namespace torch { namespace cudnn {

cudnnHandle_t getCudnnHandle();

}} // namespace
