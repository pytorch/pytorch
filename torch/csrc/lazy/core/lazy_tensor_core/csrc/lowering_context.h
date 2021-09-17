#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/ir_util.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/core/platform/macros.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {

// Keeps track of the code generation state.
class LoweringContext {
 public:
  LoweringContext(const std::string& name, Device device);
  LoweringContext(const std::string& name, Device device,
                  lazy_tensors::Span<const Node* const> post_order,
                  Util::EmissionMap emit_status);

  virtual ~LoweringContext() = default;

  static std::unique_ptr<LoweringContext> Create(
      const std::string& name, Device device,
      lazy_tensors::Span<const Node* const> post_order,
      Util::EmissionMap emit_status);

  static std::unique_ptr<LoweringContext> Create(const std::string& name,
                                                 Device device);

  const Device& device() const { return device_; };

  // Retrieves the vector holding all the tensors associated with the parameter
  // instructions which have been created.
  const std::vector<lazy_tensors::ComputationClient::DataPtr>&
  GetParametersData() const;

  // Get the shape of the result tuple component, given by index.
  virtual lazy_tensors::Shape GetResultShape(size_t index) const = 0;

  // Adds the given output as a component of the result tuple and returns its
  // assigned position within the tuple.
  virtual size_t AddResult(const Output& output) = 0;

  // Build the computation capturing all the operations created with the
  // embedded builder (returned by the builder() API).
  virtual lazy_tensors::StatusOr<
      std::shared_ptr<lazy_tensors::GenericComputation>>
  Build() = 0;

  // Lowers the given node as the result of the computation. Only used for the
  // operator-by-operator execution, mostly for debugging purposes.
  virtual void LowerNodeToResult(const Node* node);

  // Associates the given output with the input parameter of the given index and
  // shape. Only used for the operator-by-operator execution, mostly for
  // debugging purposes.
  virtual void AddParameter(const Output& output, size_t index,
                            const lazy_tensors::Shape& shape,
                            const std::string& name);

  // Indicates that the output and the parameter given by their respective
  // indices can use the same storage. The underlying back-end can safely ignore
  // this information, but it can be used to implement efficient in-place
  // operations in a semantically functional model.
  virtual void SetUpAlias(const lazy_tensors::ShapeIndex& output_index,
                          lazy_tensors::int64 param_number,
                          const lazy_tensors::ShapeIndex& param_index);

  size_t GetEmittedNodeCount() const { return emit_status_.size(); }

 protected:
  Device device_;
  std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_;
  std::vector<size_t> parameter_sequence_;
  Util::EmissionMap emit_status_;
};

}  // namespace ir
}  // namespace torch_lazy_tensors
