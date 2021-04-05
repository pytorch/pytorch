#pragma once

// include cub in a safe manner
#undef CUB_NS_POSTFIX //undef to avoid redefinition warnings
#undef CUB_NS_PREFIX
#define CUB_NS_PREFIX namespace at{ namespace native{
#define CUB_NS_POSTFIX }}
#include <cub/cub.cuh>
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
