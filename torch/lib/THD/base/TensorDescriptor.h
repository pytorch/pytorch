#pragma once

#include "../THD.h"
#include <TH/TH.h>
#ifdef WITH_CUDA
#include <THC/THC.h>
#endif

#ifndef _THD_CORE
#include <ATen/ATen.h>
using THDTensorDescriptor = at::Tensor;
#endif
