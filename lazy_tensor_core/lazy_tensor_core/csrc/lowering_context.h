#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_util.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch_lazy_tensors {

namespace compiler {

class Computation {
 public:
  virtual int parameters_size() const  = 0;

  virtual const std::vector<torch::lazy::Shape>& parameter_shapes() const = 0;

  virtual const std::vector<std::string>& parameter_names() const = 0;

  virtual const torch::lazy::Shape& result_shape() const = 0;

  virtual ~Computation() = default;
};

using ComputationPtr = std::shared_ptr<Computation>;
}

namespace ir {

// Keeps track of the code generation state.
class LoweringContext {
 public:
  LoweringContext(const std::string& name, torch::lazy::BackendDevice device);
  LoweringContext(const std::string& name, torch::lazy::BackendDevice device,
                  c10::ArrayRef<torch::lazy::Node*> post_order,
                  torch::lazy::Util::EmissionMap emit_status);

  virtual ~LoweringContext() = default;

  static std::unique_ptr<LoweringContext> Create(
      const std::string& name, torch::lazy::BackendDevice device,
      c10::ArrayRef<torch::lazy::Node*> post_order,
      torch::lazy::Util::EmissionMap emit_status);

  static std::unique_ptr<LoweringContext> Create(const std::string& name,
                                                 torch::lazy::BackendDevice device);

  const torch::lazy::BackendDevice& device() const { return device_; };

  // Retrieves the vector holding all the tensors associated with the parameter
  // instructions which have been created.
  const std::vector<torch::lazy::BackendDataPtr>& GetParametersData() const;

  // Get the shape of the result tuple component, given by index.
  virtual torch::lazy::Shape GetResultShape(size_t index) const = 0;

  // Adds the given output as a component of the result tuple and returns its
  // assigned position within the tuple.
  virtual size_t AddResult(const torch::lazy::Output& output) = 0;

  // Associates the given output with the input parameter of the given index and
  // shape. Only used for the operator-by-operator execution, mostly for
  // debugging purposes.
  virtual void AddParameter(const torch::lazy::Output& output, size_t index,
                            const torch::lazy::Shape& shape,
                            const std::string& name);

  // Build the computation capturing all the operations created with the
  // embedded builder (returned by the builder() API).
  virtual compiler::ComputationPtr Build() = 0;

  size_t GetEmittedNodeCount() const { return emit_status_.size(); }

 protected:
  torch::lazy::BackendDevice device_;
  std::vector<torch::lazy::BackendDataPtr> parameters_;
  std::vector<size_t> parameter_sequence_;
  torch::lazy::Util::EmissionMap emit_status_;
};

}  // namespace ir
}  // namespace torch_lazy_tensors
