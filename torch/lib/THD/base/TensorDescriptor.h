#pragma once

#include <TH/TH.h>
#include <THD/THD.h>
#ifdef USE_CUDA
#include <THC/THC.h>
#endif

#ifndef _THD_CORE
#include <ATen/ATen.h>
using THDTensorDescriptor = at::Tensor;
#endif
