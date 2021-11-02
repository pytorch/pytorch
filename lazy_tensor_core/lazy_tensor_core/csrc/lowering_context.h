#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lazy_tensor_core/csrc/compiler/data.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/ir_util.h" // This is already landed on master, needs rebase
#include "torch/csrc/lazy/core/ir.h"

namespace torch_lazy_tensors {

namespace compiler {

class Computation {
 public:
  virtual int parameters_size() const  = 0;

  virtual const std::vector<lazy_tensors::Shape>& parameter_shapes() const = 0;

  virtual const std::vector<std::string>& parameter_names() const = 0;

  virtual const lazy_tensors::Shape& result_shape() const = 0;

  virtual ~Computation() = default;
};

using ComputationPtr = std::shared_ptr<Computation>;
}

namespace ir {

// Keeps track of the code generation state.
class LoweringContext {
 public:
  LoweringContext(const std::string& name, Device device);
  LoweringContext(const std::string& name, Device device,
                  c10::ArrayRef<torch::lazy::Node*> post_order,
                  Util::EmissionMap emit_status);

  virtual ~LoweringContext() = default;

  static std::unique_ptr<LoweringContext> Create(
      const std::string& name, Device device,
      c10::ArrayRef<torch::lazy::Node*> post_order,
      Util::EmissionMap emit_status);

  static std::unique_ptr<LoweringContext> Create(const std::string& name,
                                                 Device device);

  const Device& device() const { return device_; };

  // Retrieves the vector holding all the tensors associated with the parameter
  // instructions which have been created.
  const std::vector<compiler::DataPtr>&
  GetParametersData() const;

  // Get the shape of the result tuple component, given by index.
  virtual lazy_tensors::Shape GetResultShape(size_t index) const = 0;

  // Adds the given output as a component of the result tuple and returns its
  // assigned position within the tuple.
  virtual size_t AddResult(const torch::lazy::Output& output) = 0;

  // Build the computation capturing all the operations created with the
  // embedded builder (returned by the builder() API).
  virtual compiler::ComputationPtr Build() = 0;

  // Lowers the given node as the result of the computation. Only used for the
  // operator-by-operator execution, mostly for debugging purposes.
  virtual void LowerNodeToResult(const torch::lazy::Node* node);

  // Associates the given output with the input parameter of the given index and
  // shape. Only used for the operator-by-operator execution, mostly for
  // debugging purposes.
  virtual void AddParameter(const torch::lazy::Output& output, size_t index,
                            const lazy_tensors::Shape& shape,
                            const std::string& name);

  size_t GetEmittedNodeCount() const { return emit_status_.size(); }

 protected:
  Device device_;
  std::vector<compiler::DataPtr> parameters_;
  std::vector<size_t> parameter_sequence_;
  Util::EmissionMap emit_status_;
};

}  // namespace ir
}  // namespace torch_lazy_tensors
