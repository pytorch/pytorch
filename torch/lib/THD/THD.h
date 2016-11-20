#pragma once

#ifdef __cplusplus__
#define THD_API extern "C"
#else
#define THD_API
#endif

#include "master_worker/master/Master.h"
#include "master_worker/master/State.h"
#include "master_worker/master/THDTensor.h"

#include "master_worker/worker/Worker.h"
