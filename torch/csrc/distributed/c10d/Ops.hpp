#pragma once

#include <c10/util/intrusive_ptr.h>
#include <c10d/ProcessGroup.hpp>

namespace c10d {
namespace ops {

// Belows are essentially ProcessGroup's corresponding ops but routed to the dispatcher.
TORCH_API c10::intrusive_ptr<ProcessGroup::Work> broadcast(const c10::intrusive_ptr<ProcessGroup>& process_group,
    at::TensorList tensors, const BroadcastOptions& opts = {});

} // namespace ops
} // namespace c10d
