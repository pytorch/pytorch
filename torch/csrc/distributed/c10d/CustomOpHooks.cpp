#include <c10/util/intrusive_ptr.h>
#include <c10d/ProcessGroup.hpp>
#include <torch/library.h>

namespace c10d {

c10::intrusive_ptr<ProcessGroup::Work> broadcast(const c10::intrusive_ptr<ProcessGroup>& process_group,
    const at::Tensor& tensor, int64_t src) {
  BroadcastOptions options;
  options.rootRank = src;
  std::vector<at::Tensor> tensors {tensor};
  return process_group->broadcast(tensors, options);
}

TORCH_LIBRARY(c10d, m) {
  m.def("broadcast", broadcast);
}

} // namespace c10d
