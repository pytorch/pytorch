// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <caffe2/torch/csrc/jit/backends/xnnpack/serialization/serializer.h>
#include <torch/csrc/jit/backends/xnnpack/serialization/schema_generated.h>

#include <sstream>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

using namespace fb_xnnpack;

void XNNSerializer::serializeAddNode(
    uint32_t input1_id,
    uint32_t input2_id,
    uint32_t output_id,
    uint32_t flags) {
  const auto addNode =
      CreateXNNAdd(_builder, input1_id, input2_id, output_id, flags);
  const auto flatbufferNode =
      CreateXNode(_builder, XNodeUnion::XNNAdd, addNode.Union());
  _nodes.push_back(flatbufferNode);
}

size_t XNNSerializer::serializeData(const uint8_t* data_ptr, size_t num_bytes) {
  size_t constant_buffer_idx = 0;
  // Handling the tensor _values with data
  if (data_ptr != nullptr) {
    // steps:
    // 1. creating flatbuffer byte-vector for tensor data
    auto storage = _builder.CreateVector(data_ptr, num_bytes);

    // 2. put it in the common buffer
    constant_buffer_idx = _constantBuffer.size();
    _constantBuffer.emplace_back(CreateBuffer(_builder, storage));

    // 3. record size into bufferSizes
    _bufferSizes.push_back(num_bytes);
    assert(_bufferSizes.size() == _constantBuffer.size());
  }
  return constant_buffer_idx;
}

void XNNSerializer::serializeTensorValue(
    uint32_t xnn_datatype,
    size_t num_dims,
    std::vector<size_t> dims,
    size_t data_buffer_idx,
    uint32_t external_id,
    uint32_t flags,
    uint32_t id_out) {
  std::vector<uint32_t> serialized_dims;
  serialized_dims.reserve(dims.size());
  for (auto dim : dims) {
    serialized_dims.push_back(static_cast<uint32_t>(dim));
  }

  const auto tensorValue = CreateXNNTensorValueDirect(
      _builder,
      XNNDatatype(xnn_datatype),
      num_dims,
      &serialized_dims,
      data_buffer_idx,
      external_id,
      flags,
      id_out);

  const auto flatbufferValue =
      CreateXValue(_builder, XValueUnion::XNNTensorValue, tensorValue.Union());
  _values.push_back(flatbufferValue);
}

std::string XNNSerializer::finishAndSerialize(
    std::vector<uint32_t> input_ids,
    std::vector<uint32_t> output_ids,
    size_t num_extern_ids) {
  auto xnnGraph = CreateXNNGraphDirect(
      _builder,
      _version_sha1,
      &_nodes,
      &_values,
      num_extern_ids,
      &input_ids,
      &output_ids,
      &_constantBuffer,
      &_bufferSizes);

  _builder.Finish(xnnGraph);

  std::stringstream ss;
  ss.write(
      reinterpret_cast<char*>(_builder.GetBufferPointer()), _builder.GetSize());

  return ss.str();
}

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
