#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <ATen/native/vulkan/api/api.h>

namespace at {
namespace native {
namespace vulkan {
namespace arithmetic {

enum class OpType : uint32_t {
  ADD,
  SUB,
  MUL,
  DIV,
  FLOOR_DIV,
  POW,
};

api::ShaderInfo get_shader(const OpType type);

void record_op(
    api::Context* const context,
    const api::ShaderInfo& compute_shader,
    vTensor& v_in1,
    vTensor& v_in2,
    vTensor& v_dst,
    const float alpha);

} // namespace arithmetic
} // namespace vulkan
} // namespace native
} // namespace at
