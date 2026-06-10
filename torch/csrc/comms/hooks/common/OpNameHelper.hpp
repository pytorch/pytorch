// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <variant>

#include <torch/csrc/comms/TorchCommHooks.hpp>

namespace torch::comms {

inline OpName getOpName(const PreHookArgs& args) {
  if (std::get_if<SendPreHookArgs>(&args))
    return OpName::send;
  else if (std::get_if<RecvPreHookArgs>(&args))
    return OpName::recv;
  else if (std::get_if<BroadcastPreHookArgs>(&args))
    return OpName::broadcast;
  else if (std::get_if<AllReducePreHookArgs>(&args))
    return OpName::all_reduce;
  else if (std::get_if<ReducePreHookArgs>(&args))
    return OpName::reduce;
  else if (std::get_if<AllGatherPreHookArgs>(&args))
    return OpName::all_gather;
  else if (std::get_if<AllGatherVPreHookArgs>(&args))
    return OpName::all_gather_v;
  else if (std::get_if<AllGatherSinglePreHookArgs>(&args))
    return OpName::all_gather_single;
  else if (std::get_if<ReduceScatterPreHookArgs>(&args))
    return OpName::reduce_scatter;
  else if (std::get_if<ReduceScatterVPreHookArgs>(&args))
    return OpName::reduce_scatter_v;
  else if (std::get_if<ReduceScatterSinglePreHookArgs>(&args))
    return OpName::reduce_scatter_single;
  else if (std::get_if<AllToAllSinglePreHookArgs>(&args))
    return OpName::all_to_all_single;
  else if (std::get_if<AllToAllVSinglePreHookArgs>(&args))
    return OpName::all_to_all_v_single;
  else if (std::get_if<AllToAllPreHookArgs>(&args))
    return OpName::all_to_all;
  else if (std::get_if<BarrierPreHookArgs>(&args))
    return OpName::barrier;
  else if (std::get_if<ScatterPreHookArgs>(&args))
    return OpName::scatter;
  else if (std::get_if<GatherPreHookArgs>(&args))
    return OpName::gather;
  else if (std::get_if<GatherSinglePreHookArgs>(&args))
    return OpName::gather_single;
  else if (std::get_if<SplitPreHookArgs>(&args))
    return OpName::split;
  else if (std::get_if<NewWindowPreHookArgs>(&args))
    return OpName::new_window;
  else if (std::get_if<BatchOpIssuePreHookArgs>(&args))
    return OpName::batch_op_issue;
  else if (std::get_if<FinalizePreHookArgs>(&args))
    return OpName::finalize;
  __builtin_unreachable();
}

} // namespace torch::comms
