// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <c10/macros/Export.h>

#include <string>
#include <unordered_map>

namespace c10d::nccl2 {

// ncclCommDump of every live (not aborted, revoked or destroyed) communicator,
// keyed by group name ("#<address>" if unnamed). Merged into the
// nccl_comm_state of the NCCL dumps.
// Defined only when (IS_NCCLX || USE_ROCM) && NCCL_COMM_DUMP.
TORCH_API std::
    unordered_map<std::string, std::unordered_map<std::string, std::string>>
    dumpNcclComms();

} // namespace c10d::nccl2
