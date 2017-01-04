#ifndef THCP_H
#define THCP_H

#include <TH/TH.h>
#include <THC/THC.h>
#include <THC/THCHalf.h>

#include <THS/THS.h>
#include <THCS/THCS.h>

#include "torch/csrc/THP.h"
#include "serialization.h"
#include "AutoGPU.h"
#include "Module.h"
#include "Storage.h"
#include "Tensor.h"
#include "Stream.h"
#ifdef _THP_CORE
#include "utils.h"
#endif

#endif
