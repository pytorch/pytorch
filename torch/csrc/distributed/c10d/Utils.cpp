#include <c10d/Utils.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace c10d {

std::string parse_env(const char* env_var_name) {
  char* stringValue = std::getenv(env_var_name);
  std::string res = "N/A";
  if (stringValue != nullptr) {
    res = stringValue;
  }
  return res;
}

std::vector<at::Tensor> getTensorShapes(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> shapeTensors;
  shapeTensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    auto shapesVec = tensor.sizes().vec();
    int64_t shapes_size = shapesVec.size();
    // Need to clone here otherwise the shapesVec.data() memory is not copied
    // and can be released under the hood.
    at::Tensor shapesTensor = at::from_blob(
                                  shapesVec.data(),
                                  {shapes_size},
                                  at::TensorOptions().dtype(at::kLong))
                                  .clone();
    shapeTensors.emplace_back(std::move(shapesTensor));
  }
  return shapeTensors;
}

} // namespace c10d
