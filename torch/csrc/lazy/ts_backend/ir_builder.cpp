#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/ts_backend/ir_builder.h>

// This file contains the TorchScript IrBuilder

namespace torch {
namespace lazy {

// Get IrBuilder from backend. Use TorchScriptIrBuilder by default
const IrBuilder* getIrBuilder() {
  static const IrBuilder* builder = hasBackend() ? getBackend()->GetIrBuilder() : new TorchScriptIrBuilder();
  return builder;
}

} // namespace lazy
} // namespace torch
