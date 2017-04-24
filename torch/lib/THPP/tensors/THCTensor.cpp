#include <THCS/THCS.h>

#include "THCTensor.hpp"
#include "THCSTensor.hpp"
#include "THCHalf.h"
#include "../TraitsCuda.hpp"

namespace thpp {

#include "generic/THCTensor.cpp"
#include <THC/THCGenerateAllTypes.h>

} // namespace thpp
