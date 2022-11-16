// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <caffe2/torch/csrc/jit/backends/xnnpack/compiler/xnn_compiler.h>
#include <torch/csrc/jit/backends/xnnpack/serialization/schema_generated.h>

#include <ATen/Utils.h>
#include <unordered_set>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

XNNExecutor XNNCompiler::compileModel(std::string ser_model) {
  const char* buffer_pointer = ser_model.data();

  auto output_min = -std::numeric_limits<float>::infinity();
  auto output_max = std::numeric_limits<float>::infinity();

  auto flatbuffer_graph = fb_xnnpack::GetXNNGraph(buffer_pointer);
  // initialize xnnpack
  xnn_status status = xnn_initialize(/*allocator =*/nullptr);
  TORCH_CHECK(xnn_status_success == status, "Failed to initialize xnnpack");

  // create xnnpack subgraph
  xnn_subgraph_t subgraph_ptr = nullptr;

  // TODO: @maxren serialize extern_ids in flatbuffer schema
  std::unordered_set<uint32_t> extern_ids;
  for (auto input_id : *flatbuffer_graph->input_ids()) {
    extern_ids.insert(input_id);
  }
  for (auto output_id : *flatbuffer_graph->output_ids()) {
    extern_ids.insert(output_id);
  }
  status = xnn_create_subgraph(
      /*external_value_ids=*/extern_ids.size(),
      /*flags=*/0,
      &subgraph_ptr);
  TORCH_CHECK(xnn_status_success == status, "Failed to create xnn subgraph");

  // mapping from old ids to new created value ids
  // The old ids that were serialied were generated AoT, since
  // we are re-defining tensor values, the defined IDs could be
  // different from the ones generated AoT, as a result, we need
  // a new mapping from the old ids to the newly created ones
  std::unordered_map<uint32_t, uint32_t> remapped_ids;

  for (auto value : *flatbuffer_graph->values()) {
    switch (value->value_type()) {
      case fb_xnnpack::ValueUnion::XNNTensorValue: {
        auto tensor_value = value->value_as_XNNTensorValue();

        const void* data_ptr = nullptr;
        auto buffer_idx = tensor_value->constant_buffer_idx();
        if (buffer_idx != 0) {
          // TODO: @maxren implement data handling
          TORCH_CHECK(false, "Constant data handling not yet implemented")
        }
        std::vector<size_t> dims_data;
        for (auto dim : *tensor_value->dims()) {
          dims_data.push_back(static_cast<size_t>(dim));
        }

        uint32_t id = XNN_INVALID_VALUE_ID;
        status = xnn_define_tensor_value(
            /*subgraph=*/subgraph_ptr,
            /*datatype=*/xnn_datatype_fp32,
            /*num_dims=*/tensor_value->num_dims(),
            /*dims=*/dims_data.data(),
            /*data=*/data_ptr,
            /*external_id=*/tensor_value->external_id(),
            /*flags=*/tensor_value->flags(),
            /*id_out=*/&id);
        TORCH_CHECK(
            status == xnn_status_success,
            "Failed to define tensor values in graph")
        // map serialized id to newly generated id
        remapped_ids.emplace(std::make_pair(tensor_value->id_out(), id));
        break;
      }
      default: {
        TORCH_CHECK(false, "Unhandled value type found in deserialization");
      }
    }
  }

  for (auto node : *flatbuffer_graph->nodes()) {
    switch (node->node_type()) {
      case fb_xnnpack::NodeUnion::XNNAdd: {
        auto graph_node = node->node_as_XNNAdd();
        status = xnn_define_add2(
            subgraph_ptr,
            output_min,
            output_max,
            remapped_ids.at(graph_node->input1_id()),
            remapped_ids.at(graph_node->input2_id()),
            remapped_ids.at(graph_node->output_id()),
            graph_node->flags());
        TORCH_CHECK(status == xnn_status_success, "Failed to create add node")
        break;
      }
      default:
        TORCH_CHECK(false, "Unhandled node type found in deserialization");
    }
  }

  xnn_runtime_t runtime_ptr = nullptr;
  status = xnn_create_runtime_v2(subgraph_ptr, nullptr, 0, &runtime_ptr);
  TORCH_CHECK(xnn_status_success == status);

  XNNExecutor executor(runtime_ptr);

  for (auto old_id : *flatbuffer_graph->input_ids()) {
    executor.input_ids_.push_back(remapped_ids.at(old_id));
  }

  for (auto old_id : *flatbuffer_graph->output_ids()) {
    executor.output_ids_.push_back(remapped_ids.at(old_id));
  }

  return executor;
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
