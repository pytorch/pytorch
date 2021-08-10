#include <ATen/AccumulateType.h>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/NativeFunctions.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/accumulate.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>
#include <torch/library.h>

#include <ATen/FunctionalTensorImpl.h>
#include <ATen/RedispatchFunctions.h>

#include <algorithm>
#include <cstdint>
#include <vector>

// ${generated_comment}

namespace at {
namespace functionalization {


${func_definitions}
}  // namespace func

namespace {

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  ${func_registrations};
}

}  // namespace

} // namespace at
