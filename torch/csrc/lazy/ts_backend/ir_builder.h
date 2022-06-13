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
  NodePtr MakeView(const Value& input0, const std::vector<int64_t>& output_size)
      const override {
    return ReuseOrMakeNode<View>(input0, output_size);
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

  // View op nodes
  NodePtr MakeAsStridedViewUpdate(
      const Value& input0,
      const Value& input1,
      const std::vector<int64_t>& size,
      const std::vector<int64_t>& stride,
      const int64_t& storage_offset) const override {
    return ReuseOrMakeNode<AsStridedViewUpdate>(
        input0, input1, size, stride, storage_offset);
  }
  NodePtr MakeAsStrided(
      const Value& input0,
      const std::vector<int64_t>& size,
      const std::vector<int64_t>& stride,
      const int64_t& storage_offset) const override {
    return ReuseOrMakeNode<AsStrided>(input0, size, stride, storage_offset);
  }
  NodePtr MakeDiagonalViewUpdate(
      const Value& input0,
      const Value& input1,
      const int64_t& offset,
      const int64_t& dim1,
      const int64_t& dim2) const override {
    return ReuseOrMakeNode<DiagonalViewUpdate>(
        input0, input1, offset, dim1, dim2);
  }
  NodePtr MakeDiagonal(
      const Value& input0,
      const int64_t& offset,
      const int64_t& dim1,
      const int64_t& dim2) const override {
    return ReuseOrMakeNode<Diagonal>(input0, offset, dim1, dim2);
  }
  NodePtr MakeNarrowViewUpdate(
      const Value& input0,
      const Value& input1,
      const std::vector<int64_t>& base_indices) const override {
    return ReuseOrMakeNode<NarrowViewUpdate>(input0, input1, base_indices);
  }
  NodePtr MakeNarrow(
      const Value& input0,
      const std::vector<int64_t>& base_indices,
      const std::vector<int64_t>& sizes) const override {
    return ReuseOrMakeNode<Narrow>(input0, base_indices, sizes);
  }
  NodePtr MakePermute(const Value& input0, const std::vector<int64_t>& dims)
      const override {
    return ReuseOrMakeNode<Permute>(input0, dims);
  }
  NodePtr MakeResize(const Value& input0, const std::vector<int64_t>& size)
      const override {
    return ReuseOrMakeNode<Resize>(input0, size);
  }
  NodePtr MakeSelectViewUpdate(
      const Value& input0,
      const Value& input1,
      const int64_t& dim,
      const int64_t& start,
      const int64_t& end,
      const int64_t& stride) const override {
    return ReuseOrMakeNode<SelectViewUpdate>(
        input0, input1, dim, start, end, stride);
  }
  NodePtr MakeSelect(
      const Value& input0,
      const int64_t& dim,
      const int64_t& start,
      const int64_t& end,
      const int64_t& stride) const override {
    return ReuseOrMakeNode<Select>(input0, dim, start, end, stride);
  }
  NodePtr MakeSqueeze(const Value& input0, const int& dim) const override {
    return ReuseOrMakeNode<Squeeze>(input0, dim);
  }
  NodePtr MakeUnsqueeze(const Value& input0, const int& dim) const override {
    return ReuseOrMakeNode<Unsqueeze>(input0, dim);
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
