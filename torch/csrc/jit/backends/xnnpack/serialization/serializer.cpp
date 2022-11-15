// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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

void XNNSerializer::serializeTensorValue(
    uint32_t xnn_datatype,
    size_t num_dims,
    std::vector<size_t> dims,
    void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t id_out) {
  // we will reserve buffers without data to index 0
  int constant_buffer_idx = 0;
  // Handling the tensor _values with data
  // TODO @maxren fill out when handling tensors with data
  if (data != nullptr) {
    assert(false); // not supported yet
    // steps:
    // 1. creating buffer to store the 16 bit aligned data
    // 2. increment buffer_idx, to reflect no buffer being added
    // 3. record size into bufferSizes
  }

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
      constant_buffer_idx,
      external_id,
      flags,
      id_out);

  const auto flatbufferValue =
      CreateXValue(_builder, XValueUnion::XNNTensorValue, tensorValue.Union());
  _values.push_back(flatbufferValue);
}

std::string XNNSerializer::finishAndSerialize(
    std::vector<uint32_t> input_ids,
    std::vector<uint32_t> output_ids) {
  auto xnnGraph = CreateXNNGraphDirect(
      _builder,
      _version_sha1,
      &_nodes,
      &_values,
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
