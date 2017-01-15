#include "THDTensor.h"
#include "base/Traits.hpp"
#include "base/tensors/THTensor.hpp"
#include "State.hpp"
#include "master_worker/common/RPC.hpp"
#include "master_worker/common/Functions.hpp"
#include "master_worker/master/Master.hpp"
#include "process_group/General.hpp"

#include <memory>

#include "master_worker/master/generic/THDTensor.cpp"
#include "TH/THGenerateAllTypes.h"
