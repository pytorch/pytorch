#include <torch/nn/options/linear.h>

namespace torch {
namespace nn {

LinearOptions::LinearOptions(int64_t in_features, int64_t out_features)
 : in_features_(in_features), out_features_(out_features) {}

BilinearOptions::BilinearOptions(int64_t in1_features, int64_t in2_features, int64_t out_features)
  : in1_features_(in1_features), in2_features_(in2_features), out_features_(out_features) {}

UnflattenOptions::UnflattenOptions(int64_t dim, std::vector<int64_t> unflattened_size)
  : dim_(dim), unflattened_size_(unflattened_size) {}

UnflattenOptions::UnflattenOptions(std::string dim, namedshape_t unflattened_size)
  : dim_(dim), unflattened_size_(unflattened_size) {}

} // namespace nn
} // namespace torch
