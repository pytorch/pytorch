#pragma once

#include <c10/util/intrusive_ptr.h>
#include <c10d/ProcessGroup.hpp>

namespace c10d {
namespace ops {

// Below are essentially ProcessGroup's corresponding ops but routed to the dispatcher.
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> broadcast(const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors, const BroadcastOptions& opts = {});
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> allreduce(const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors, const AllreduceOptions& opts = {});

} // namespace ops
} // namespace c10d
