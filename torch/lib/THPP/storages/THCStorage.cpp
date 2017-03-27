#include "THCStorage.hpp"
#include "../Traits.hpp"

#include "../tensors/THCTensor.hpp"

namespace thpp {

#include "generic/THCStorage.cpp"
#include <THC/THCGenerateAllTypes.h>

} // namespace thpp
