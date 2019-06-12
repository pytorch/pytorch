#ifndef THCP_H
#define THCP_H

#include "torch/csrc/python_headers.h"
#include <TH/TH.h>
#include <THC/THC.h>
#include <TH/THHalf.h>
#include <THC/THCTensor.hpp>

#include "torch/csrc/THP.h"
#include "serialization.h"
#include "Module.h"
#include "Storage.h"
#include "Stream.h"
#ifdef _THP_CORE
#include "utils.h"
#endif

#endif
