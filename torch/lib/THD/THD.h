#pragma once

#ifdef __cplusplus
#define THD_API extern "C"
#else
#define THD_API
#endif

#ifndef _THD_CORE
#include "base/DataChannelRequest.h"
#include "base/TensorDescriptor.h"
#else
#include "base/DataChannelRequest.hpp"
#include "base/TensorDescriptor.hpp"
#endif
#include "base/ChannelType.h"
#include "base/Cuda.h"

#include "process_group/Collectives.h"
#include "process_group/General.h"
