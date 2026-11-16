#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch::jit {

struct ChunkOutput {
  ChunkOutput(Value* v, size_t o) : val(v), offset(o) {}
  Value* val;
  size_t offset;
};

static std::optional<std::vector<ChunkOutput>> getChunkOutputs(Node* chunk) {
  std::vector<ChunkOutput> outputs;
  for (auto list_use : chunk->output()->uses()) {
    if (list_use.user->matches(
            "aten::select(t[] list, int idx) -> t", attr::idx) &&
        list_use.user->output()->type()->cast<TensorType>()) {
      outputs.emplace_back(
          list_use.user->output(),
          list_use.user->get<int64_t>(attr::idx).value());
    } else if (list_use.user->kind() == prim::ListUnpack) {
      // This sometimes happens if the sizes can't be evenly divided by the
      // number of chunks
      if (static_cast<int64_t>(list_use.user->outputs().size()) !=
          chunk->get<int64_t>(attr::chunks).value()) {
        return std::nullopt;
      }
      auto unpack_outputs = list_use.user->outputs();
      for (const auto i : c10::irange(unpack_outputs.size())) {
        outputs.emplace_back(unpack_outputs[i], i);
      }
    } else {
      return std::nullopt;
    }
  }
  return outputs;
}

static void CanonicalizeOps(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks())
      CanonicalizeOps(sub);
    if (it->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        it->matches(
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        it->matches("aten::mul(Tensor self, Tensor other) -> Tensor") ||
        it->matches("aten::div(Tensor self, Tensor other) -> Tensor")) {
      // Replace rank 0 Tensor constants with scalar constants.
      if (auto other = it->get<at::Tensor>(attr::other)) {
        if (other->dim() == 0) {
          WithInsertPoint insert_guard{*it};
          auto graph = it->owningGraph();
          auto new_other = graph->insertConstant(other->item());
          std::vector<Value*> inputs = it->inputs().vec();
          inputs.at(1) = new_other;
          Value* new_output =
              graph->insertNode(graph->create(it->kind(), inputs))->output();
          new_output->node()->copyMetadata(*it);
          new_output->copyMetadata(it->output());
          it->output()->replaceAllUsesWith(new_output);
        }
      }
    } else if (it->matches(
                   "aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]",
                   /*const_inputs=*/{attr::chunks, attr::dim})) {
      // Replace aten::chunk (which returns a list) with ConstantChunk with the
      // outputs unpacked.
      if (auto orig_outputs = getChunkOutputs(*it)) {
        WithInsertPoint guard(*it);
        auto* self = it->namedInput(attr::self);
        auto* graph = it->owningGraph();
        const auto chunks = it->get<int64_t>(attr::chunks).value();
        const auto dim = it->get<int64_t>(attr::dim).value();
        auto* node =
            graph->insertNode(graph->create(prim::ConstantChunk, chunks));
        node->addInput(self);
        node->i_(attr::chunks, chunks)->i_(attr::dim, dim);
        node->copyMetadata(*it);
        for (const auto& orig_out : *orig_outputs) {
          orig_out.val->replaceAllUsesWith(node->outputs()[orig_out.offset]);
          node->outputs()[orig_out.offset]->setType(orig_out.val->type());
        }
      }
    }
  }
}

void CanonicalizeOps(const std::shared_ptr<Graph>& graph) {
  CanonicalizeOps(graph->block());
  GRAPH_DUMP("After CanonicalizeOps: ", graph);
  EliminateDeadCode(graph);
}

} // namespace torch::jit
