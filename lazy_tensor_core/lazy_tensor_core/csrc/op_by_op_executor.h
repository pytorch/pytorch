#pragma once

#include <string>
#include <vector>

#include "lazy_tensors/computation_client/async_task.h"
#include "lazy_tensors/computation_client/cache.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"
#include "torch/csrc/lazy/core/ir.h"

namespace torch_lazy_tensors {

// The OpByOpExecutor class is a singleton accessible via its Get() API that
// allows to run an IR graph is per-IR-node isolation mode. Instead of lowering
// the whole IR graph in a single computation, the single IR nodes are lowered
// and executed independently.
class OpByOpExecutor {
 public:
  using AsyncResult = std::vector<lazy_tensors::ComputationClient::DataPtr>;
  using AsyncTask = lazy_tensors::util::AsyncTask<AsyncResult>;

  static OpByOpExecutor* Get();

  std::vector<lazy_tensors::ComputationClient::ExecuteChainedOp> BuildOps(
      lazy_tensors::Span<const torch::lazy::Value> roots, const std::string& device,
      lazy_tensors::Span<const std::string> devices);

  std::vector<lazy_tensors::ComputationClient::DataPtr> Execute(
      lazy_tensors::Span<const torch::lazy::Value> roots, const std::string& device,
      lazy_tensors::Span<const std::string> devices);

  AsyncTask ExecuteAsync(lazy_tensors::Span<const torch::lazy::Value> roots,
                         const std::string& device,
                         lazy_tensors::Span<const std::string> devices);

 private:
  using CompileCache =
      lazy_tensors::util::Cache<torch::lazy::hash_t,
                                lazy_tensors::ComputationClient::Computation,
                                torch::lazy::HashReducer>;

  explicit OpByOpExecutor(size_t compile_cache_size);

  CompileCache compile_cache_;
};

}  // namespace torch_lazy_tensors
