#pragma once

#include <memory>
#include <string>
#include <vector>

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_util.h>

namespace torch::lazy {

class TORCH_API Computation {
 public:
  virtual int parameters_size() const = 0;

  virtual const std::vector<Shape>& parameter_shapes() const = 0;

  virtual const std::vector<std::string>& parameter_names() const = 0;

  virtual const Shape& result_shape() const = 0;

  virtual const std::string to_string() const = 0;

  virtual ~Computation() = default;

  // Indicates whether this computation is being executed inside a mark step
  // Assume false unless set otherwise
  bool in_mark_step = false;
};

using ComputationPtr = std::shared_ptr<Computation>;

// Keeps track of the code generation state.
class TORCH_API LoweringContext {
 public:
  LoweringContext(const std::string& name, BackendDevice device);
  LoweringContext(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      Util::EmissionMap emit_status);

  virtual ~LoweringContext() = default;

  static std::unique_ptr<LoweringContext> Create(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      Util::EmissionMap emit_status);

  static std::unique_ptr<LoweringContext> Create(
      const std::string& name,
      BackendDevice device);

  const BackendDevice& device() const {
    return device_;
  }

  // Retrieves the vector holding all the tensors associated with the parameter
  // instructions which have been created.
  const std::vector<BackendDataPtr>& GetParametersData() const;

  // Adds a new input/output alias.
  virtual void SetUpAlias(
      const std::vector<int64_t>& output_index,
      int64_t param_number,
      const std::vector<int64_t>& param_index,
      bool must_alias = false) {
    // Dummy default implementation to do nothing.
  }

  // Check if parameter shape matches result at index.
  virtual bool CheckResultShape(
      const BackendDataPtr& parameter_data,
      size_t result_idx) {
    // Dummy default implementation to do nothing.
    return false;
  }

  // Adds the given output as a component of the result tuple and returns its
  // assigned position within the tuple.
  virtual size_t AddResult(const torch::lazy::Output& output) = 0;

  // Associates the given output with the input parameter of the given index and
  // shape. Only used for the operator-by-operator execution, mostly for
  // debugging purposes.
  virtual void AddParameter(
      const torch::lazy::Output& output,
      size_t index,
      const Shape& shape,
      const std::string& name) = 0;

  // Build the computation capturing all the operations created with the
  // embedded builder (returned by the builder() API).
  virtual ComputationPtr Build() = 0;

  size_t GetEmittedNodeCount() const {
    return emit_status_.size();
  }

 protected:
  BackendDevice device_;
  std::vector<BackendDataPtr> parameters_;
  std::vector<size_t> parameter_sequence_;
  Util::EmissionMap emit_status_;
};

} // namespace torch::lazy
