#ifndef THDP_H
#define THDP_H

#include <THD/THD.h>

#include "torch/csrc/THP.h"
#include "Module.h"
#ifdef WITH_DISTRIBUTED_MW
#include "Storage.h"
#include "../PtrWrapper.h"
#ifdef _THP_CORE
#include "utils.h"
#endif
#endif

#endif
