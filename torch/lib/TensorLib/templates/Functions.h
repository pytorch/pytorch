#pragma once

#include "TensorLib/Scalar.h"
#include "TensorLib/Type.h"
#include "TensorLib/Tensor.h"
#include "TensorLib/Storage.h"
#include "TensorLib/Generator.h"



namespace tlib {

${function_declarations}

// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument
${function_definitions}

}
