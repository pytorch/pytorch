#pragma once

#ifdef __cplusplus
#define THD_API extern "C"
#else
#define THD_API
#endif

#ifndef _THD_CORE
#include "base/TensorDescriptor.h"
#else
#include "base/TensorDescriptor.hpp"
#endif
#include "base/ChannelType.h"

#include "process_group/General.h"
#include "process_group/Collectives.h"

#include "master_worker/master/Master.h"
#include "master_worker/master/State.h"
#include "master_worker/master/THDTensor.h"

#include "master_worker/worker/Worker.h"
