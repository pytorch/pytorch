#pragma once
// Pass: decode_op_signatures (KernelProgram -> DecodedOps, read-only).
//
// Recognizes structured eager externs whose target is exactly one of
//   aten.mm.default, aten.addmm.default, aten.bmm.default,
//   aten.sum.dim_IntList, aten.mean.dim
// and binds their typed CallArg args/kwargs against the ATen schema
// (positional + keyword, keyword-only enforced, defaults applied). Tensor
// operands take sizes/strides from the extern's own descriptors; physical
// placement (storage root + composed view/descriptor offset) is recorded
// separately in TensorBinding. The original ExternCall is kept verbatim as
// the fallback. Signature operand names are the IR names (a repeated
// argument gets a "#slot" suffix; TensorBinding.ir_name stays exact).
// Unknown targets are recorded as UnknownTarget and never
// interpreted. A recognized target whose ABI does not match the schema is
// MalformedAbi (fail closed: no signature is used); well-formed calls no
// option can implement (non-f32, negative strides, dim=[]...) are
// Unsupported with a typed reason.

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "../kernel_ir.h"
#include "op_types.h"

namespace torch::inductor::loops_ir::op_options {

enum class DecodeStatus { Decoded, UnknownTarget, Unsupported, MalformedAbi };
const char* decodeStatusName(DecodeStatus status);

// Physical operand placement, separate from the logical TensorLayout: the
// kernel/native tensor is root.storage_offset() + root_offset with the
// operand's sizes/strides.
// Precondition (checked by decode): 0 <= root_offset and root_offset +
// layout extent <= root_elements. node_result: the root is a program node
// result (a fresh internal allocation), not a program input that a caller
// could alias with another argument.
struct TensorBinding {
  std::string ir_name;
  std::string root;
  int64_t root_offset = 0;
  int64_t root_elements = 0;
  bool node_result = false;
};

struct DecodedOp {
  explicit DecodedOp(ExternCall call) : fallback(std::move(call)) {}
  size_t node_index = 0;
  DecodeStatus status = DecodeStatus::UnknownTarget;
  std::optional<RejectReason> reason;
  std::string detail;
  std::optional<OpSignature> signature;
  std::vector<TensorBinding> input_bindings; // per signature input slot
  std::optional<TensorBinding> output_binding;
  ExternCall fallback;
};

struct DecodedOps {
  std::string numerics; // KernelProgram::numerics()
  std::vector<DecodedOp> ops; // one per ExternCall node, program order
};

DecodedOps decodeOpSignatures(const KernelProgram& program, DeviceKind device);

} // namespace torch::inductor::loops_ir::op_options
