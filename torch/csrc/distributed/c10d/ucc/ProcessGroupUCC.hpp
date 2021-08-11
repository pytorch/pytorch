#pragma once

#include <c10d/Store.hpp>
#include <c10d/ProcessGroup.hpp>

namespace c10d {

typedef c10::intrusive_ptr<ProcessGroup> CreateProcessGroupUCCType(const c10::intrusive_ptr<Store>& store, int rank, int size);

} // namespace c10d
