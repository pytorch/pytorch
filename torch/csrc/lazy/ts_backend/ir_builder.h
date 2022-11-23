#pragma once

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/shape_inference.h>
#include <torch/csrc/lazy/generated/LazyNonNativeIr.h>
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>
#include <torch/csrc/lazy/ts_backend/ops/device_data.h>
#include <torch/csrc/lazy/ts_backend/ops/generic.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

struct TorchScriptIrBuilder : IrBuilder {
  NodePtr MakeDeviceData(
      const std::shared_ptr<BackendData>& data) const override {
    return DeviceData::Create(data);
  }
  // TODO: Scalar node is not currently used by ts_backend. Enable reusing
  // Scalar node later if needed.
  NodePtr MakeScalar(const at::Scalar& value, const at::ScalarType& type)
      const override {
    return MakeNode<Scalar>(value, type);
  }
  NodePtr MakeExpand(
      const Value& input0,
      const std::vector<int64_t>& size,
      const bool& is_scalar_expand) const override {
    return ReuseOrMakeNode<Expand>(input0, size, is_scalar_expand);
  }
  NodePtr MakeCast(
      const Value& input0,
      const at::ScalarType& dtype,
      const c10::optional<at::ScalarType>& stype =
          c10::nullopt) const override {
    return ReuseOrMakeNode<Cast>(input0, dtype, stype);
  }
  NodePtr MakeTensorList(const OpList& inputs) const override {
    return ReuseOrMakeNode<TensorList>(inputs);
  }
  // Generic needs cleanup
  NodePtr MakeGeneric(
      const OpKind& op,
      const OpList& operands,
      const Shape& shape,
      const size_t& num_outputs = 1,
      const hash_t& hash_seed =
          static_cast<uint32_t>(0x5a2d296e9)) const override {
    return MakeNode<Generic>(op, operands, shape, num_outputs, hash_seed);
  }

  // dynamic ir nodes
  // TODO: verify if IR node reusing works for Dynamic shape ops
  NodePtr MakeSizeNode(const Value& input, size_t dim) const override {
    return MakeNode<SizeNode>(input, dim);
  }
  NodePtr MakeSizeAdd(const Value& a, const Value& b) const override {
    return MakeNode<SizeAdd>(a, b);
  }
  NodePtr MakeSizeMul(const Value& a, const Value& b) const override {
    return MakeNode<SizeMul>(a, b);
  }
  NodePtr MakeSizeDiv(const Value& a, const Value& b) const override {
    return MakeNode<SizeDiv>(a, b);
  }
};

} // namespace lazy
} // namespace torch
