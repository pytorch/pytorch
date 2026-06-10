// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/hooks/common/OpNameHelper.hpp>
#include <torch/csrc/comms/hooks/common/SignatureBuilder.hpp>

#include <variant>

#include <fmt/core.h>

namespace torch::comms {

std::string_view dtypeToString(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return "f32";
    case at::ScalarType::Double:
      return "f64";
    case at::ScalarType::Half:
      return "f16";
    case at::ScalarType::BFloat16:
      return "bf16";
    case at::ScalarType::Int:
      return "i32";
    case at::ScalarType::Long:
      return "i64";
    case at::ScalarType::Short:
      return "i16";
    case at::ScalarType::Char:
      return "i8";
    case at::ScalarType::Byte:
      return "u8";
    case at::ScalarType::Bool:
      return "bool";
    case at::ScalarType::Float8_e4m3fn:
      return "fp8e4m3";
    case at::ScalarType::Float8_e5m2:
      return "fp8e5m2";
    default:
      return "other";
  }
}

std::string_view reduceOpToString(const ReduceOp& op) {
  switch (op.type()) {
    case ReduceOp::RedOpType::SUM:
      return "sum";
    case ReduceOp::RedOpType::PRODUCT:
      return "prod";
    case ReduceOp::RedOpType::MIN:
      return "min";
    case ReduceOp::RedOpType::MAX:
      return "max";
    case ReduceOp::RedOpType::BAND:
      return "band";
    case ReduceOp::RedOpType::BOR:
      return "bor";
    case ReduceOp::RedOpType::BXOR:
      return "bxor";
    case ReduceOp::RedOpType::PREMUL_SUM:
      return "premul_sum";
    case ReduceOp::RedOpType::AVG:
      return "avg";
    default:
      return "unknown";
  }
}

std::string formatPtr(const void* ptr) {
  return fmt::format("{:#x}", reinterpret_cast<uintptr_t>(ptr));
}

std::string formatPtrs(const std::vector<at::Tensor>& tensors) {
  std::string result;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (i > 0) {
      result += ',';
    }
    result += formatPtr(tensors[i].data_ptr());
  }
  return result;
}

std::string formatCounts(const std::vector<at::Tensor>& tensors) {
  std::string result;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (i > 0) {
      result += ',';
    }
    result += std::to_string(tensors[i].numel());
  }
  return result;
}

std::string formatCounts(const std::vector<uint64_t>& counts) {
  std::string result;
  for (size_t i = 0; i < counts.size(); ++i) {
    if (i > 0) {
      result += ',';
    }
    result += std::to_string(counts[i]);
  }
  return result;
}

std::string buildSignature(
    std::string_view comm_name,
    const PreHookArgs& args,
    bool include_buffers) {
  auto op = opToString(getOpName(args));
  auto sig = std::visit(
      [include_buffers, op](const auto& a) -> std::string {
        using T = std::decay_t<decltype(a)>;

        if constexpr (std::is_same_v<T, AllReducePreHookArgs>) {
          auto count = a.tensor.numel();
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto red_op = reduceOpToString(a.op);
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|red_op={}|async_op={}",
              op,
              count,
              count,
              dtype,
              red_op,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format("|buf={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, BroadcastPreHookArgs>) {
          auto count = a.tensor.numel();
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|root={}|async_op={}",
              op,
              count,
              count,
              dtype,
              a.root,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format("|buf={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, ReducePreHookArgs>) {
          auto count = a.tensor.numel();
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto red_op = reduceOpToString(a.op);
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|red_op={}|root={}|async_op={}",
              op,
              count,
              count,
              dtype,
              red_op,
              a.root,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format("|buf={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, SendPreHookArgs>) {
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_count=0|dtype={}|peer={}|async_op={}",
              op,
              a.tensor.numel(),
              dtype,
              a.peer,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format("|in={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, RecvPreHookArgs>) {
          auto dtype = dtypeToString(a.tensor.scalar_type());
          auto sig = fmt::format(
              "{}|in_count=0|out_count={}|dtype={}|peer={}|async_op={}",
              op,
              a.tensor.numel(),
              dtype,
              a.peer,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format("|out={}", formatPtr(a.tensor.data_ptr()));
          }
          return sig;
        } else if constexpr (
            std::is_same_v<T, AllGatherSinglePreHookArgs> ||
            std::is_same_v<T, AllToAllSinglePreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|async_op={}",
              op,
              a.input.numel(),
              a.output.numel(),
              dtype,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtr(a.output.data_ptr()));
          }
          return sig;
        } else if constexpr (std::
                                 is_same_v<T, ReduceScatterSinglePreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto red_op = reduceOpToString(a.op);
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|red_op={}|async_op={}",
              op,
              a.input.numel(),
              a.output.numel(),
              dtype,
              red_op,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtr(a.output.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, AllToAllVSinglePreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_splits={}|out_splits={}|dtype={}|async_op={}",
              op,
              formatCounts(a.input_split_sizes),
              formatCounts(a.output_split_sizes),
              dtype,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtr(a.output.data_ptr()));
          }
          return sig;
        } else if constexpr (
            std::is_same_v<T, AllGatherPreHookArgs> ||
            std::is_same_v<T, AllGatherVPreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_counts={}|dtype={}|async_op={}",
              op,
              a.input.numel(),
              formatCounts(a.output),
              dtype,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtrs(a.output));
          }
          return sig;
        } else if constexpr (
            std::is_same_v<T, ReduceScatterPreHookArgs> ||
            std::is_same_v<T, ReduceScatterVPreHookArgs>) {
          auto dtype = dtypeToString(a.output.scalar_type());
          auto red_op = reduceOpToString(a.op);
          auto sig = fmt::format(
              "{}|in_counts={}|out_count={}|dtype={}|red_op={}|async_op={}",
              op,
              formatCounts(a.input),
              a.output.numel(),
              dtype,
              red_op,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtrs(a.input),
                formatPtr(a.output.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, AllToAllPreHookArgs>) {
          if (a.input.empty()) {
            return std::string();
          }
          auto dtype = dtypeToString(a.input[0].scalar_type());
          auto sig = fmt::format(
              "{}|in_counts={}|out_counts={}|dtype={}|async_op={}",
              op,
              formatCounts(a.input),
              formatCounts(a.output),
              dtype,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format(
                "|in={}|out={}", formatPtrs(a.input), formatPtrs(a.output));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, ScatterPreHookArgs>) {
          auto dtype = dtypeToString(a.output.scalar_type());
          auto sig = fmt::format(
              "{}|in_counts={}|out_count={}|dtype={}|root={}|async_op={}",
              op,
              formatCounts(a.input),
              a.output.numel(),
              dtype,
              a.root,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtrs(a.input),
                formatPtr(a.output.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, GatherPreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_counts={}|dtype={}|root={}|async_op={}",
              op,
              a.input.numel(),
              formatCounts(a.output),
              dtype,
              a.root,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtrs(a.output));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, GatherSinglePreHookArgs>) {
          auto dtype = dtypeToString(a.input.scalar_type());
          auto sig = fmt::format(
              "{}|in_count={}|out_count={}|dtype={}|root={}|async_op={}",
              op,
              a.input.numel(),
              a.output.numel(),
              dtype,
              a.root,
              a.async_op ? 't' : 'f');
          if (include_buffers) {
            sig += fmt::format(
                "|in={}|out={}",
                formatPtr(a.input.data_ptr()),
                formatPtr(a.output.data_ptr()));
          }
          return sig;
        } else if constexpr (std::is_same_v<T, BarrierPreHookArgs>) {
          return fmt::format("{}|async_op={}", op, a.async_op ? 't' : 'f');
        } else if constexpr (std::is_same_v<T, BatchOpIssuePreHookArgs>) {
          return fmt::format(
              "{}|num_ops={}|async_op={}",
              op,
              a.num_ops,
              a.async_op ? 't' : 'f');
        } else if constexpr (
            std::is_same_v<T, SplitPreHookArgs> ||
            std::is_same_v<T, NewWindowPreHookArgs> ||
            std::is_same_v<T, FinalizePreHookArgs>) {
          return std::string();
        } else {
          return std::string();
        }
      },
      args);
  if (!sig.empty()) {
    sig += fmt::format("|comm={}", comm_name);
  }
  return sig;
}

std::string buildSplitLine(
    std::string_view parent_name,
    const SplitPreHookArgs& split) {
  std::string ranks_str;
  for (size_t i = 0; i < split.ranks.size(); ++i) {
    if (i > 0) {
      ranks_str += ',';
    }
    ranks_str += std::to_string(split.ranks[i]);
  }
  return fmt::format(
      "split|parent={}|child={}|ranks={}", parent_name, split.name, ranks_str);
}

std::string buildNewCommSignature(
    std::string_view comm_name,
    int rank,
    int world_size) {
  return fmt::format(
      "new_comm|comm={}|rank={}|world_size={}", comm_name, rank, world_size);
}

} // namespace torch::comms
