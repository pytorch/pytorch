#pragma once

#ifdef __cplusplus
#define THD_API extern "C"
#else
#define THD_API
#endif

#ifndef _THD_CORE
#include <THD/base/DataChannelRequest.h>
#include <THD/base/TensorDescriptor.h>
#else
#include <THD/base/DataChannelRequest.hpp>
#include <THD/base/TensorDescriptor.hpp>
#endif
#include <THD/base/ChannelType.h>
#include <THD/base/Cuda.h>

#include <THD/process_group/Collectives.h>
#include <THD/process_group/General.h>
