#pragma once

#include <ATen/Dimname.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct Storage;
}

namespace at {

class Tensor;
using TensorList = ArrayRef<Tensor>;

class Context;
struct Generator;

struct Quantizer;

} // namespace at
