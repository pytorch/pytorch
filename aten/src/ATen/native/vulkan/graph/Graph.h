#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Tensor.h>

#include <ATen/native/vulkan/graph/Config.h>
#include <ATen/native/vulkan/graph/Exception.h>
#include <ATen/native/vulkan/graph/Value.h>

namespace at {
namespace native {
namespace vulkan {

typedef int32_t ValueRef;
class ComputeGraph;

class OpNode {
  friend class ComputeGraph;

 public:
  virtual ~OpNode() {}

 protected:
  std::vector<ValueRef> inputs_;
  std::vector<ValueRef> outputs_;

 public:
  virtual void encode(ComputeGraph* graph) {}
};

class ComputeGraph final {
 public:
  explicit ComputeGraph(GraphConfig config);

  ComputeGraph(ComputeGraph&&) = default;
  ComputeGraph& operator=(ComputeGraph&&) = default;

  ~ComputeGraph();

 private:
  GraphConfig config_;
  std::unique_ptr<api::Context> context_;
  std::vector<Value> values_;
  std::vector<std::unique_ptr<OpNode>> nodes_;

  std::vector<ValueRef> inputs_;
  std::vector<ValueRef> outputs_;

 public:
  //
  // Accessors
  //

  inline api::Context* context() {
    return context_.get();
  }

  /*
   * Returns the value at a particular reference
   */
  inline Value& get_val(ValueRef idx) {
    return values_[idx];
  }

  /*
   * Looks up the value at a particular reference.
   * 1. If it's a Tensor, return it as a tensor
   * 2. If it's a staging, return the tensor member of the staging struct
   * 3. Otherwise throw an error
   */
  inline vTensor& tensor_at(ValueRef idx) {
    Value& val = get_val(idx);
    if (val.isTensor()) {
      return val.toTensor();
    } else if (val.isStaging()) {
      return val.toStaging().tensor;
    }
    VKGRAPH_THROW("Expected value to be Tensor or Staging!");
  }

  inline std::vector<std::unique_ptr<OpNode>>& nodes() {
    return nodes_;
  }

  //
  // Graph Building
  //

  void add_input_tensor(IntArrayRef& sizes, c10::ScalarType dtype);
  void add_output_tensor(IntArrayRef& sizes, c10::ScalarType dtype);

  void add_node(OpNode& node);

  //
  // Graph Execution
  //

  void encode();
  void execute();

  //
  // Input/Output
  //

  void copy_into_input(ValueRef idx, void* data);
  void copy_from_output(ValueRef idx, void* data);
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
