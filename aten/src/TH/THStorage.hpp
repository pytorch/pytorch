#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THStorage.h"

#include "ATen/ScalarType.h"
#include "ATen/ScalarTypeUtils.h"
#include "THTypeConversion.hpp"
#include <atomic>

#include "generic/THStorage.hpp"
#include "THGenerateAllTypes.h"

#include "generic/THStorage.hpp"
#include "THGenerateHalfType.h"
