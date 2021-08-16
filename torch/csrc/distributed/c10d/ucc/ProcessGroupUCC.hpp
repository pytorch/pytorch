#pragma once

// This header is designed to be includible even if PyTorch is built with UCC
// disabled. Please don't use any symbol from UCX, UCC, or libtorch_ucc.so.

#include <c10d/Store.hpp>
#include <c10d/ProcessGroup.hpp>

namespace c10d {

typedef c10::intrusive_ptr<ProcessGroup> (*CreateProcessGroupUCCType)(
    const c10::intrusive_ptr<Store>& store, int rank, int size);

} // namespace c10d
