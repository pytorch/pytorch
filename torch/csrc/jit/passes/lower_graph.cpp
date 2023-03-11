#include <torch/csrc/jit/passes/lower_graph.h>

#include <torch/csrc/jit/api/object.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/custom_class.h>
#include <unordered_map>

namespace torch {
namespace jit {

struct Slot {
  c10::intrusive_ptr<c10::ivalue::Object> obj;
  size_t offset;
  bool operator==(const Slot& other) const {
    return (this->obj == other.obj && this->offset == other.offset);
  }
};

// remove the first module argument, replacing any access of its
// parameters/attributes with extra_ivalue input Slots that hold what value to
// pass into the graph. Used for ONNX export to remove first-class modules
// so it can deal purely with parameters and inputs
std::pair<std::shared_ptr<Graph>, std::vector<Slot>> lower_graph(
    const ModulePtr& self,
    Graph& g_,
    size_t self_offset = 0) {
  std::shared_ptr<Graph> g = g_.copy();
  // Inline to remove method/function calls
  Inline(*g);

  std::vector<Slot> extra_ivalues;

  struct SlotHash {
    std::size_t operator()(const Slot& slot) const {
      auto obj_hash = std::hash<c10::ivalue::Object*>{}(slot.obj.get());
      auto offset_hash = std::hash<size_t>{}(slot.offset);
      return c10::hash_combine(obj_hash, offset_hash);
    }
  };
  std::unordered_map<Slot, size_t, SlotHash> slot_to_offset;
  struct ToScan {
    ModulePtr mod;
    Node* n;
    size_t offset;
  };
  std::vector<ToScan> to_scan;
  std::vector<Node*> to_clean; // nodes that should be dead at the end

  auto getOrAddSlot = [&](const Slot& slot) -> Value* {
    auto it = slot_to_offset.find(slot);
    if (it != slot_to_offset.end()) {
      size_t ivalues_start = g->inputs().size() - extra_ivalues.size();
      return g->inputs().at(ivalues_start + it->second);
    }
    extra_ivalues.emplace_back(slot);
    slot_to_offset[slot] = extra_ivalues.size() - 1;
    return g->addInput()->setType(slot.obj->getSlot(slot.offset).type());
  };

  auto self_value = g->inputs().at(self_offset);

  for (Use use : self_value->uses()) {
    to_scan.emplace_back(ToScan{self, use.user, use.offset});
  }
  while (!to_scan.empty()) {
    auto e = to_scan.back();
    to_scan.pop_back();

    // when we lambda lift forks, first-class modules may be passed across
    // forks. This code recursively lowers the module in the fork call.
    if (e.n->kind() == prim::fork) {
      auto subgraph = e.n->g(attr::Subgraph);
      std::vector<Slot> new_slots;
      std::tie(subgraph, new_slots) = lower_graph(e.mod, *subgraph, e.offset);
      e.n->g_(attr::Subgraph, subgraph);
      for (const Slot& slot : new_slots) {
        e.n->addInput(getOrAddSlot(slot));
      }
      e.n->removeInput(e.offset);
      continue;
    }
    if (e.n->kind() == prim::PythonOp) {
      throw ErrorReport(e.n->sourceRange()) << "Couldn't export Python method.";
    }
    if (e.n->kind() != prim::GetAttr) {
      throw ErrorReport(e.n->sourceRange())
          << "temporary: the only valid use of a module is looking up an "
             "attribute but found "
          << *e.n;
    }
    size_t slot_idx = e.mod->type()->getAttributeSlot(e.n->s(attr::name));
    auto iv = e.mod->getSlot(slot_idx);
    if (ClassTypePtr c = e.n->output()->type()->cast<ClassType>()) {
      if (c->is_module()) {
        for (Use use : e.n->output()->uses()) {
          to_scan.emplace_back(ToScan{iv.toObject(), use.user, use.offset});
        }
        to_clean.emplace_back(e.n);
        continue;
      }
    }
    e.n->output()->replaceAllUsesWith(getOrAddSlot({e.mod, slot_idx}));
    e.n->destroy();
  }

  while (!to_clean.empty()) {
    Node* n = to_clean.back();
    AT_ASSERT(!n->hasUses());
    n->destroy();
    to_clean.pop_back();
  }
  AT_ASSERT(!self_value->hasUses());
  g->eraseInput(self_offset);

  return std::make_pair(std::move(g), std::move(extra_ivalues));
}

static std::vector<IValue> loadTensors(const std::vector<Slot>& slots) {
  std::vector<IValue> result;
  result.reserve(slots.size());
  for (const Slot& slot : slots) {
    auto obj = slot.obj->getSlot(slot.offset);
    if (obj.isTensor()) {
      result.emplace_back(obj.toTensor());
    } else {
      // Unpack quantization packed tensor
      auto type = obj.type();
      TORCH_CHECK(
          (type ==
           getCustomClass(
               "__torch__.torch.classes.quantized.Conv2dPackedParamsBase")) ||
              (type ==
               getCustomClass(
                   "__torch__.torch.classes.quantized.Conv3dPackedParamsBase")) ||
              (type ==
               getCustomClass(
                   "__torch__.torch.classes.quantized.LinearPackedParamsBase")),
          "Unknown type ",
          type->repr_str(),
          " encountered in graph lowering. This type is not supported in ONNX export.");
      result.emplace_back(
          script::Object(obj.toObject()).run_method("__getstate__"));
    }
  }
  return result;
}

std::pair<std::shared_ptr<Graph>, std::vector<IValue>> LowerGraph(
    Graph& graph,
    const ModulePtr& self) {
  auto result = lower_graph(self, graph);
  return std::make_pair(result.first, loadTensors(result.second));
}

} // namespace jit
} // namespace torch
