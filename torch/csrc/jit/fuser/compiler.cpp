#include "torch/csrc/jit/fuser/compiler.h"

#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/type.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_cache.h"

#include <iostream>
#include <memory>
#include <unordered_set>
#include <utility>
#include <string>
#include <atomic>
#include <sstream>
#include <stdexcept>

namespace torch { namespace jit { namespace fuser {

static std::atomic<size_t> next_kernel_id{0};

void registerFusion(int64_t& key, const Node* fusion_group) {

  // Creates and stores the FusionSpec
  auto graph = fusion_group->g(attr::Subgraph)->copy();
  EraseShapeInformation(*graph);
  key = store(graph);

// Creates and stores the fusion spec
// const auto maybe_spec = retrieve(key);
// if (!maybe_spec) {
//   // TODO: error out
// // }
// // Validates the graph
// bool is_fusable = true;
// for (const auto& node : graph->nodes()) {
// if (!::torch::jit::fuser::isSupportedOp(node)) {
// std::cout << node->kind().toDisplayString() << std::endl;
// is_fusable = false;
// break;
// }
// }
// if (!is_fusable) {
// const auto maybe_spec = retrieve(key);
// if (!maybe_spec) 
// throw std::runtime_error("Registered fusion specification not found.");
// maybe_spec->setFusable(false);
// return ReturnCode::UNSUPPORTED_OP;
// }

// // Performs device-independent upfront compilation of the spec 
// // if (canFuseOnCPU() || canFuseOnGPU())
// //   upfrontCompilation(*maybe_spec);
}
} // namespace fuser
} // namespace jit
} // namespace torch