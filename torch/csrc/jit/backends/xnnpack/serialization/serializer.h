// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/csrc/jit/backends/xnnpack/serialization/schema_generated.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

using namespace fb_xnnpack; // Specified in the schema

class XNNSerializer {
 public:
  // Constructors
  // initial buffersize of 1024 which will grow
  // automatically, constant buffer and buffer sizes initialized with dummy
  // values as 0 index is reserved for non-constant tensors
  XNNSerializer() : XNNSerializer(1024) {}

  explicit XNNSerializer(size_t bufferSize)
      : _builder(bufferSize),
        _nodes(),
        _values(),
        _constantBuffer({CreateBuffer(
            _builder,
            {})}), // index 0 is reserved for non-const data
        _bufferSizes({0}) {}

  // Serializing Nodes

  // Serialize add node, we are serializing the argument needed to call
  // xnn_define_add2. Serializing these values, and at run time we build
  // teh graph by re running xnn_define_add2
  void serializeAddNode(
      uint32_t input1_id,
      uint32_t input2_id,
      uint32_t output_id,
      uint32_t flags);

  // Serializing Values
  void serializeTensorValue(
      uint32_t xnn_datatype,
      size_t num_dims,
      std::vector<size_t> dims,
      size_t buffer_data_idx,
      uint32_t external_id,
      uint32_t flags,
      uint32_t id_out);

  // finish and serialize xnngraph returning serialized data
  std::string finishAndSerialize(
      std::vector<uint32_t> input_ids,
      std::vector<uint32_t> output_ids,
      size_t num_extern_ids);

  // decoupled data serialization with tensor values. This way constant tensor
  // data can be referenced by multiple intermediate tensors. This call
  // serializes the num_bytes of the data_ptr and returns the index it was
  // placed in.
  size_t serializeData(const uint8_t* data_ptr, size_t num_bytes);

 private:
  // xnnpack version we are serializing
  const char* _version_sha1 = "ae108ef49aa5623b896fc93d4298c49d1750d9ba";

  // flatbuffer objects we will create and serialize together to create xnngraph
  flatbuffers_fbsource::FlatBufferBuilder _builder;

  // Vector of the serialized xnnpack nodes
  std::vector<flatbuffers_fbsource::Offset<XNode>> _nodes;

  // Vector of the serialized xnnpack values
  std::vector<flatbuffers_fbsource::Offset<XValue>> _values;

  std::vector<flatbuffers_fbsource::Offset<Buffer>> _constantBuffer;
  std::vector<uint32_t> _bufferSizes;
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
