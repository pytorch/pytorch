#include <c10/util/intrusive_ptr.h>
#include <c10d/ProcessGroup.hpp>
#include <torch/library.h>

namespace c10d {

c10::intrusive_ptr<ProcessGroup::Work> broadcast(const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors, int64_t root_rank = 0, int64_t root_tensor = 0, int64_t timeout = -1) {
  auto tensor_vec = tensors.vec();
  return process_group->broadcast(tensor_vec,
      BroadcastOptions {root_rank, root_tensor, std::chrono::milliseconds(timeout)});
}

TORCH_LIBRARY(c10d, m) {
  m.class_<ProcessGroup>("ProcessGroup")
    .def(torch::init<int64_t, int64_t>());
  m.class_<ProcessGroup::Work>("Work")
    .def(torch::init<>());
  m.def("broadcast", broadcast);
}

} // namespace c10d
