#include <torch/csrc/jit/passes/onnx/nezha_helper.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <c10/util/Optional.h>
#include <algorithm>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

void NeZha_TryUpdateGraph(
    std::shared_ptr<Graph>& dst_graph,
    std::shared_ptr<Graph>& src_graph) {
    
    printf("------ Start NeZha_TryUpdateGraph ------");
    dst_graph = std::make_shared<Graph>();
    auto env = [](Value* v) -> Value* {
    AT_ERROR(
        "Graph::copy() encountered a use of a value " + v->debugName() +
        " not in scope. Run lint!");
    };
    printf("------ NeZha_TryUpdateGraph: Clone block to dst_graph ------");
    dst_graph->block()->cloneFrom(src_graph->block(), env);
}


} // namespace jit
} // namespace torch
