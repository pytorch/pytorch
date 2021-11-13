#include "lazy_tensor_core/csrc/data_ops.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <ATen/InferSize.h>
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {



std::vector<int64_t> BuildUnsqueezeDimensions(c10::ArrayRef<int64_t> dimensions,
                                              int64_t dim) {
  CHECK_LE(dim, dimensions.size());
  auto unsqueeze_dimensions = lazy_tensors::util::ToVector<int64_t>(dimensions);
  unsqueeze_dimensions.insert(unsqueeze_dimensions.begin() + dim, 1);
  return unsqueeze_dimensions;
}

}  // namespace torch_lazy_tensors
