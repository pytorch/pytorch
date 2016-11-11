#pragma once

#ifdef __cplusplus__
#define THD_API extern "C"
#else
#define THD_API
#endif

#include "master/Master.h"
#include "master/State.h"
#include "master/THDTensor.h"

#include "worker/Worker.h"
