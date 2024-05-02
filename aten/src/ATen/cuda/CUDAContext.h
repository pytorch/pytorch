#pragma once

#include <ATen/cuda/CUDAContextLight.h>

// Preserved for BC, as many files depend on these includes
#include <ATen/Context.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Logging.h>
#include <ATen/cuda/Exceptions.h>
