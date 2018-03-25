#include "torch/csrc/assertions.h"

#include <ATen/ATen.h>
#include <unordered_map>

namespace torch { namespace cuda {

using tensor_list2d = std::vector<std::vector<at::Tensor>>;

std::vector<at::Tensor> broadcast(const at::Tensor& tensor, at::IntList devices);
tensor_list2d broadcast_coalesced(at::TensorList tensors, at::IntList devices,
                                  std::size_t buffer_size);

}}
