// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <caffe2/torch/csrc/jit/backends/xnnpack/xnnpack_graph_builder.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <xnnpack.h>

// graph passes
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

std::shared_ptr<torch::jit::Graph> XNNGraph::optimizeAndTraceGraph(
    std::shared_ptr<torch::jit::Graph> graph,
    std::vector<c10::IValue>& example_inputs) {
  OptimizeFrozenGraph(graph, true);
  RemoveListMutation(graph);
  RemoveTensorMutation(graph);
  LowerAllTuples(graph);
  ConstantPropagation(graph);
  graph = TraceGraph(graph, example_inputs);

  return graph;
}

void XNNGraph::buildXNNGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    std::vector<c10::IValue> example_inputs) {
  graph = optimizeAndTraceGraph(graph, example_inputs);
  checkOpsToDelegate(graph);
  gatherTensorValues(graph);

  // count unique input/outputs (some inputs can be outputs)
  std::unordered_set<torch::jit::Value*> externals;
  for (auto inp : _inputs) {
    externals.insert(inp);
  }
  for (auto out : _outputs) {
    externals.insert(out);
  }

  // create subgraph
  xnn_status status = xnn_create_subgraph(
      /*external_value_ids=*/externals.size(),
      /*flags=*/0,
      &_subgraph_ptr);
  TORCH_CHECK(xnn_status_success == status, "Failed to create xnn subgraph");

  defineAllTensorValues();
  defineAllNodes(graph);
  // at this point graph is complete, for the sake of testing preprocess at
  // this point we will do runtime setup and run with some default values
}

void XNNGraph::runGraphOnInputs(
    std::vector<at::Tensor> tensor_inputs,
    std::vector<at::Tensor> tensor_outputs) {
  TORCH_CHECK(
      _subgraph_ptr != nullptr,
      "run buildXNNGraph before running graph on inputs");
  xnn_runtime_t runtime = nullptr;
  xnn_status status =
      xnn_create_runtime_v2(_subgraph_ptr, nullptr, /*flags=*/0, &runtime);
  TORCH_CHECK(
      xnn_status_success == status,
      "failed to create runtime for running inputs");

  // smart pointer for runtime
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  std::vector<xnn_external_value> external_values;
  TORCH_CHECK(
      tensor_inputs.size() == _inputs.size(),
      "supplied inputs does not match expected inputs");
  for (int i = 0; i < tensor_inputs.size(); i++) {
    external_values.push_back(
        {_val_to_ids[_inputs[i]], tensor_inputs[i].data_ptr<float>()});
  }

  TORCH_CHECK(
      tensor_outputs.size() == _outputs.size(),
      "supplied outputs does not match expected outputs");
  for (int i = 0; i < tensor_outputs.size(); i++) {
    external_values.push_back(
        {_val_to_ids[_outputs[i]], tensor_outputs[i].data_ptr<float>()});
  }
  status = xnn_setup_runtime(
      auto_runtime.get(), external_values.size(), external_values.data());
  TORCH_CHECK(xnn_status_success == status, "runtime not properly setup");

  TORCH_CHECK(xnn_status_success == xnn_invoke_runtime(auto_runtime.get()));
}

void XNNGraph::checkOpsToDelegate(std::shared_ptr<torch::jit::Graph>& graph) {
  std::unordered_set<string> unsupported_ops;
  DepthFirstGraphNodeIterator it(graph);
  Node* node = nullptr;
  while ((node = it.next()) != nullptr) {
    switch (node->kind()) {
      case prim::Constant:
      case aten::add: {
        break;
      }
      default: {
        unsupported_ops.insert(node->kind().toDisplayString());
      }
    }
  }
  std::stringstream error;
  for (auto itr = unsupported_ops.begin(); itr != unsupported_ops.end();
       itr++) {
    error << *itr << std::endl;
    ;
  }
  TORCH_CHECK(
      unsupported_ops.empty(),
      "the module contains the following unsupported ops:\n" + error.str());
}

std::string XNNGraph::serializedXNNGraph() {
  std::vector<uint32_t> input_ids;
  std::vector<uint32_t> output_ids;
  std::unordered_set<uint32_t> num_externs;

  for (auto val : _inputs) {
    input_ids.push_back(_val_to_ids[val]);
    num_externs.emplace(_val_to_ids[val]);
  }

  for (auto val : _outputs) {
    output_ids.push_back(_val_to_ids[val]);
    num_externs.emplace(_val_to_ids[val]);
  }

  return _serializer.finishAndSerialize(
      input_ids, output_ids, num_externs.size());
}

std::vector<std::vector<long>> XNNGraph::getGraphOutputShapes() {
  std::vector<std::vector<long>> output_shapes;
  for (auto val : _outputs) {
    auto tensor_ptr = val->type()->cast<TensorType>();
    std::vector<long> sizes = tensor_ptr->sizes().concrete_sizes().value();
    output_shapes.push_back(sizes);
  }

  return output_shapes;
}

void XNNGraph::defineAllNodes(std::shared_ptr<torch::jit::Graph>& graph) {
  DepthFirstGraphNodeIterator it(graph);
  Node* node = nullptr;
  while ((node = it.next()) != nullptr) {
    switch (node->kind()) {
      case prim::Constant: {
        break;
      }
      case aten::add: {
        // todo: handle alpha for aten::add
        uint32_t input1_id = _val_to_ids[node->inputs()[0]];
        uint32_t input2_id = _val_to_ids[node->inputs()[1]];
        TORCH_CHECK(
            node->inputs()[2]->type()->cast<IntType>() == 1,
            "non-1 alpha values not supported");
        uint32_t output_id = _val_to_ids[node->outputs()[0]];

        xnn_status status = xnn_define_add2(
            _subgraph_ptr,
            output_min,
            output_max,
            input1_id,
            input2_id,
            output_id,
            /*flags=*/0);
        _serializer.serializeAddNode(input1_id, input2_id, output_id, 0);
        TORCH_CHECK(status == xnn_status_success, "failed to create add node");
        break;
      }
      default: {
        throw std::exception();
        TORCH_CHECK(
            false,
            "The node of ",
            node->kind().toQualString(),
            " is not supported yet");
        break;
      }
    }
  }
}

void XNNGraph::defineAllTensorValues() {
  uint32_t external_id =
      std::numeric_limits<decltype(XNN_INVALID_VALUE_ID)>::min();
  for (auto val : _intermediate_tensors) {
    if (_val_to_ids.find(val) == _val_to_ids.end()) {
      uint32_t id = XNN_INVALID_VALUE_ID;

      // cast value to tensortype
      auto tensor_ptr = val->type()->cast<TensorType>();
      auto num_dims = tensor_ptr->dim().value();

      // create size_t* for tensor shape, casting must be done from long ->
      // size_t
      std::vector<long> sizes = tensor_ptr->sizes().concrete_sizes().value();
      std::vector<size_t> tensor_shape;
      tensor_shape.reserve(sizes.size());
      for (auto dim : sizes) {
        TORCH_CHECK(dim >= 0, "Input Dims should be unsigned");
        tensor_shape.push_back(static_cast<size_t>(dim));
      }

      // ext_id value
      uint32_t ext_id = XNN_INVALID_VALUE_ID;

      // update flag for if tensor is either graph input/output
      uint32_t flags = 0;

      // Check if value was produced by prim::Constant
      void* value_data = nullptr;
      size_t buffer_idx = 0;
      size_t num_bytes = 0;
      if (val->node()->kind() == prim::Constant) {
        c10::optional<IValue> constant = val->node()->t(attr::value);
        auto const_val = constant->toIValue().toTensor();
        // Need tensor data to be contiguous for serialization
        auto cont_const_val = const_val.contiguous();
        value_data = cont_const_val.data_ptr();

        num_bytes = const_val.storage().nbytes();
        buffer_idx = _serializer.serializeData(
            static_cast<const uint8_t*>(value_data), num_bytes);
      }

      if (isGraphInput(val) || isGraphOutput(val)) {
        if (isGraphInput(val)) {
          flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
        }
        if (isGraphOutput(val)) {
          flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
        }
        ext_id = external_id++;
      }
      xnn_status status = xnn_define_tensor_value(
          /*subgraph=*/_subgraph_ptr,
          /*datatype=*/xnn_datatype_fp32,
          /*num_dims=*/num_dims,
          /*dims=*/tensor_shape.data(),
          /*data=*/value_data,
          /*external_id=*/ext_id,
          /*flags=*/flags,
          /*id_out=*/&id);
      TORCH_CHECK(
          status == xnn_status_success,
          "failed to define xnn_tensor_id for: " + val->debugName());
      _serializer.serializeTensorValue(
          xnn_datatype_fp32,
          num_dims,
          tensor_shape,
          buffer_idx,
          ext_id,
          flags,
          id);
      _val_to_ids.insert({val, id});
    }
  }
}

void XNNGraph::gatherTensorValues(std::shared_ptr<torch::jit::Graph>& graph) {
  for (auto input : graph->inputs()) {
    if (input->isCompleteTensor()) {
      _intermediate_tensors.insert(input);
      _inputs.push_back(input);
    }
  }

  DepthFirstGraphNodeIterator it(graph);
  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    gatherNodeInputs(*n);
  }

  for (auto output : graph->outputs()) {
    if (output->isCompleteTensor()) {
      _intermediate_tensors.insert(output);
      _outputs.push_back(output);
    }
  }
}

void XNNGraph::gatherNodeInputs(torch::jit::Node& node) {
  switch (node.kind()) {
    case aten::add: {
      // this case will support all ops with only two inputs i.e. sub, add,
      for (auto value : node.inputs()) {
        if (value->isCompleteTensor()) {
          _intermediate_tensors.insert(value);
        }
      }
    }
  }
}

bool XNNGraph::isGraphInput(torch::jit::Value* val) {
  return std::count(_inputs.begin(), _inputs.end(), val) > 0;
};

bool XNNGraph::isGraphOutput(torch::jit::Value* val) {
  return std::count(_outputs.begin(), _outputs.end(), val) > 0;
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
