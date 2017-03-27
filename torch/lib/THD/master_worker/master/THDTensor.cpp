#include "THDTensor.h"
#include "State.hpp"
#include "master_worker/common/RPC.hpp"
#include "master_worker/common/Functions.hpp"
#include "master_worker/master/Master.hpp"
#include "process_group/General.hpp"

#include <THPP/Traits.hpp>

#include <memory>

#include "master_worker/master/generic/THDTensor.cpp"
#include "TH/THGenerateAllTypes.h"

#include "master_worker/master/generic/THDTensorCopy.cpp"
#include "TH/THGenerateAllTypes.h"

#include "master_worker/master/generic/THDTensorMath.cpp"
#include "TH/THGenerateAllTypes.h"
