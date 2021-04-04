#pragma once

// include cub in a safe manner
#define CUB_NS_PREFIX namespace at{ namespace native{
#define CUB_NS_POSTFIX }}
#include <cub/cub.cuh>
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
