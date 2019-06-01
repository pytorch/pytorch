#include <torch/csrc/jit/passes/canonicalize_ops.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/symbolic_variable.h>

namespace torch {
namespace jit {

struct ChunkOutput {
  ChunkOutput(Value* v, size_t o) : val(v), offset(o){};
  Value* val;
  size_t offset;
};

static c10::optional<std::vector<ChunkOutput>> getChunkOutputs(Node* chunk) {
  std::vector<ChunkOutput> outputs;
  for (auto list_use : chunk->output()->uses()) {
    if (list_use.user->matches(
            "aten::select(Tensor[] list, int idx) -> Tensor", attr::idx)) {
      outputs.emplace_back(
          list_use.user->output(),
          list_use.user->get<int64_t>(attr::idx).value());
    } else if (list_use.user->kind() == prim::ListUnpack) {
      // This sometimes happens if the sizes can't be evenly divided by the
      // number of chunks
      if (static_cast<int64_t>(list_use.user->outputs().size()) !=
          chunk->get<int64_t>(attr::chunks).value()) {
        return c10::nullopt;
      }
      auto unpack_outputs = list_use.user->outputs();
      for (size_t i = 0; i < unpack_outputs.size(); ++i) {
        outputs.emplace_back(unpack_outputs[i], i);
      }
    } else {
      return c10::nullopt;
    }
  }
  return outputs;
}

static void CanonicalizeOps(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks())
      CanonicalizeOps(sub);
    if (
        it->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        it->matches(
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        it->matches("aten::mul(Tensor self, Tensor other) -> Tensor") ||
        it->matches("aten::div(Tensor self, Tensor other) -> Tensor")) {
      if (auto other = it->get<at::Tensor>(attr::other)) {
        if (other->dim() == 0) {
          WithInsertPoint insert_guard{*it};
          auto graph = it->owningGraph();
          auto new_other = graph->insertConstant(other->item());
          std::vector<Value*> inputs = it->inputs().vec();
          inputs.at(1) = new_other;
          Value* new_output =
              graph->insertNode(graph->create(it->kind(), inputs))->output();
          it->output()->replaceAllUsesWith(new_output);
        }
      }
    } else if (it->matches(
                   "aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]",
                   /*const_inputs=*/{attr::chunks, attr::dim})) {
      if (auto orig_outputs = getChunkOutputs(*it)) {
        WithInsertPoint guard(*it);
        SymbolicVariable self{it->namedInput(attr::self)};
        auto outputs = self.chunk(
            it->get<int64_t>(attr::chunks).value(),
            it->get<int64_t>(attr::dim).value());
        for (ChunkOutput orig_out : *orig_outputs) {
          orig_out.val->replaceAllUsesWith(outputs.at(orig_out.offset));
          outputs[orig_out.offset].value()->setType(orig_out.val->type());
        }
      }
    }
  }
}

void CanonicalizeOps(const std::shared_ptr<Graph>& graph) {
  CanonicalizeOps(graph->block());
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
