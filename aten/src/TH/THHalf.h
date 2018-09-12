#ifndef TH_HALF_H
#define TH_HALF_H

#ifdef __cplusplus
#include <ATen/core/Half.h>
#endif

#ifdef __cplusplus
#define THHalf at::Half
#else
typedef struct at_Half at_Half;
#define THHalf at_Half
#endif

#endif
