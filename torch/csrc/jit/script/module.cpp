#include <torch/csrc/jit/script/module.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {
namespace script {

struct RecursiveMethodCallError : public std::exception {};
void placeholderCreator(Function&) {
  throw RecursiveMethodCallError();
}

void Function::ensure_defined() {
  try {
    if (function_creator_) {
      auto creator = function_creator_;
      function_creator_ = placeholderCreator;
      creator(*this);
      function_creator_ = nullptr;
    }
  } catch (RecursiveMethodCallError&) {
    throw ErrorReport() // TODO: once lower_first_class methods is removed
                        // re-establish callsite info for debugging
        << " method '" << name()
        << "' is called recursively. "
        << "Recursive calls are not supported";
  }
}

Value* Function::try_emit_call(
    Graph& graph,
    const SourceRange& loc,
    c10::optional<NamedValue> self,
    ArrayRef<NamedValue> args,
    ArrayRef<NamedValue> kwargs,
    std::stringstream& failure_messages,
    bool conv_tensors_to_nums) {
  ensure_defined();
  auto fn = this->graph();

  auto matched_schema = tryMatchSchema(
      getSchema(),
      loc,
      graph,
      std::move(self),
      args,
      kwargs,
      failure_messages,
      conv_tensors_to_nums);
  if (!matched_schema)
    return nullptr;

  check_single_output();
  return inlineCallTo(graph, *fn, matched_schema->inputs).at(0);
}

Value* Function::emit_call(
    Graph& graph,
    const SourceRange& loc,
    ArrayRef<NamedValue> args,
    ArrayRef<NamedValue> kwargs) {
  std::stringstream failure_messages;
  if (auto result = try_emit_call(
          graph,
          loc,
          c10::nullopt,
          args,
          kwargs,
          failure_messages,
          /*conv_tensors_to_nums=*/true)) {
    return result;
  }
  throw ErrorReport(loc) << failure_messages.str();
}

void Module::to(at::Device device, at::ScalarType dtype, bool non_blocking) {
  to_impl(device, dtype, non_blocking);
}

void Module::to(at::ScalarType dtype, bool non_blocking) {
  to_impl(/*device=*/c10::nullopt, dtype, non_blocking);
}

void Module::to(at::Device device, bool non_blocking) {
  to_impl(device, /*dtype=*/c10::nullopt, non_blocking);
}

void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) {
  ExportModule(*this, out, extra_files);
}

void Module::save(
    const std::string& filename,
    const ExtraFilesMap& extra_files) {
  ExportModule(*this, filename, extra_files);
}

void Module::to_impl(
    const c10::optional<at::Device>& device,
    const c10::optional<at::ScalarType>& dtype,
    bool non_blocking) {
  // First call `to()` on every child module.
  for (auto& child : get_modules()) {
    child->to_impl(device, dtype, non_blocking);
  }
  // Then convert every of our parameters.
  for (auto& parameter : get_parameters()) {
    // Need to access the `at::Tensor` as a `Variable` here.
    autograd::Variable variable = parameter.value().toTensor();
    at::Tensor data = variable.data();
    // Use the data's original device or dtype if not supplied here.
    auto new_data = data.to(
        device.value_or(data.device()),
        dtype.value_or(data.scalar_type()),
        non_blocking);
    variable.set_data(new_data);
  }
}


// lower_first_class_method and lift_lowered_method are transitionary functions
// used to translate between module-as-first-class code generation,
// and module-as-special execution. Once module-as-first-class execution is
// debugged, then we can remove both and remove the lowered_functions_ table.

// remove the first module argument, replacing any access of its parameters/attributes
// with extra_ivalue input Slots that hold what value to pass into the graph
std::pair<std::shared_ptr<Graph>, std::vector<Slot>> lower_graph(
    const ModulePtr& self,
    Graph& g_,
    size_t self_offset = 0) {
  std::shared_ptr<Graph> g = g_.copy();
  std::vector<Slot> extra_ivalues;
  std::unordered_map<Slot, size_t> slot_to_offset;
  struct ToScan {
    ModulePtr mod;
    Node * n;
    size_t offset;
  };
  std::vector<ToScan> to_scan;
  std::vector<Node*> to_clean; // nodes that should be dead at the end
  std::vector<Node*> forks_edited; // we need to potentially de-dup inputs
                                   // to the fork node

  auto getOrAddSlot = [&](const Slot& slot) -> Value* {
    auto it = slot_to_offset.find(slot);
    if (it != slot_to_offset.end()) {
      size_t ivalues_start = g->inputs().size() - extra_ivalues.size();
      return g->inputs().at(ivalues_start + it->second);
    }
    extra_ivalues.emplace_back(slot);
    slot_to_offset[slot] = extra_ivalues.size() - 1;
    return g->addInput()->setType(slot.type());
  };


  auto self_value = g->inputs().at(self_offset);

  for (Use use : self_value->uses()) {
    to_scan.emplace_back(ToScan{self, use.user, use.offset});
  }
  while (to_scan.size() > 0) {
    auto e = to_scan.back();
    to_scan.pop_back();

    // when we lambda lift forks, first-class modules may be passed across
    // forks. This code recursively lowers the module in the fork call.
    if (e.n->kind() == prim::fork) {
      auto subgraph = e.n->g(attr::Subgraph);
      std::vector<Slot> new_slots;
      std::tie(subgraph, new_slots) = lower_graph(e.mod, *subgraph, e.offset);
      e.n->g_(attr::Subgraph, subgraph);
      for(const Slot& slot : new_slots) {
        e.n->addInput(getOrAddSlot(slot));
      }
      e.n->removeInput(e.offset);
      forks_edited.emplace_back(e.n);
      continue;
    }
    if (e.n->kind() != prim::GetAttr) {
      throw ErrorReport(e.n->getSourceLocation())
          << "temporary: the only valid use of a module is looking up an attribute";
    }
    Slot slot(e.mod, e.mod->type()->getAttributeSlot(e.n->s(attr::name)));
    if (ClassTypePtr c = e.n->output()->type()->cast<ClassType>()) {
      if (c->name() == "Module") {
        auto obj = slot.value().toObject();
        for (Use use : e.n->output()->uses()) {
          to_scan.emplace_back(ToScan{obj, use.user, use.offset});
        }
        to_clean.emplace_back(e.n);
        continue;
      }
    }
    e.n->output()->replaceAllUsesWith(getOrAddSlot(slot));
    e.n->destroy();
  }

  while (to_clean.size() > 0) {
    Node* n = to_clean.back();
    AT_ASSERT(!n->hasUses());
    n->destroy();
    to_clean.pop_back();
  }
  AT_ASSERT(!self_value->hasUses());
  g->eraseInput(self_offset);

  // for(Node* fork : forks_edited) {
  //   std::unordered_map<Value*, size_t> input_to_offset;
  //   auto subgraph = fork->g(attr::Subgraph);
  //   for(size_t i = 0; i < fork->inputs().size();) {
  //     Value* input = fork->inputs().at(i);
  //     auto it = input_to_offset.find(input);
  //     if (it == input_to_offset.end()) {
  //       input_to_offset[input] = i;
  //       ++i;
  //       continue;
  //     }
  //     // remove duplicate for input at offset i with first use at offset it->second
  //     fork->removeInput(i);
  //     subgraph->inputs().at(i)->replaceAllUsesWith(
  //         subgraph->inputs().at(it->second));
  //     subgraph->eraseInput(i);
  //   }
  // }

  return std::make_pair(std::move(g), std::move(extra_ivalues));
}

Method& Module::lower_first_class_method(Function* fn) {
  fn->ensure_defined();
  auto lowered = lower_graph(module_object(), *fn->graph());
  Function& new_func = lowered_methods_.create_function(fn->name(), lowered.first);

  // generate the new schema
  // slice away the self argument
  std::vector<Argument> args(fn->getSchema().arguments().begin() + 1, fn->getSchema().arguments().end());
  size_t id = 0;
  for (const Slot& slot : lowered.second) {
    std::ostringstream ss;
    ss << "slot" << id++;
    args.emplace_back(ss.str(), slot.type());
  }
  new_func.setSchema(fn->getSchema().withArguments(std::move(args)));
  return _create_lowered_method(&new_func, std::move(lowered.second));
}


static void createFirstClassValues(Module* module, Value* self, std::unordered_map<Slot, Value*>& result) {
  auto& g = *self->owningGraph();

  std::vector<Node*> created;
  struct ToScan {
    Module* mod;
    Value* v; // value representing module in the graph
  };
  std::vector<ToScan> to_scan = { {module,self} };

  while (!to_scan.empty()) {
    auto s = to_scan.back();
    to_scan.pop_back();
    size_t offset = 0;
    for (const std::string& name : s.mod->module_object()->type()->attributeNames()) {
      Value* v = g.insertGetAttr(s.v, name);
      result[Slot(s.mod->module_object(), offset++)] = v;
      if (std::shared_ptr<Module> sub = s.mod->find_module(name)) {
        to_scan.emplace_back(ToScan{sub.get(), v});
      }
    }
  }
}

void Module::lift_lowered_method(Method& m) {
  auto graph = m.graph()->copy();
  Value* self = graph->insertInput(0, "self")->setType(module_object()->type());
  std::unordered_map<Slot, Value*> slot_to_value;
  if (!m.initial_ivalues().empty()) {
    WithInsertPoint guard(*graph->nodes().begin());
    createFirstClassValues(this, self, slot_to_value);
  }

  size_t orig_graph_inputs_size = graph->inputs().size();
  for (size_t i = 0; i < m.initial_ivalues().size(); ++i) {
    size_t input_offset = orig_graph_inputs_size - i - 1;
    size_t ivalue_offset = m.initial_ivalues().size() - i - 1;
    graph->inputs()
        .at(input_offset)
        ->replaceAllUsesWith(
            slot_to_value.at(m.initial_ivalues().at(ivalue_offset)));
    graph->eraseInput(input_offset);
  }

  if (!m.initial_ivalues().empty()) {
    // we added _all_ the submodules as first-class values but maybe did not use
    // them. So remove any dead attribute lookups
    EliminateDeadCode(graph);
  }

  Function& new_fn = class_cu().create_function(m.name(), std::move(graph));
  // created lifted schema
  std::vector<Argument> new_args = {Argument("_self", module_object()->type())};
  const auto& lowered_args = m.function().getSchema().arguments();
  new_args.insert(new_args.end(), lowered_args.begin(), lowered_args.begin() + m.num_inputs());
  new_fn.setSchema(m.function().getSchema().withArguments(std::move(new_args)));
}

Method& Module::_create_lowered_method(
    Function* func,
    std::vector<Slot> member_inputs) {
  std::unique_ptr<Method> m(new Method(
      this, func, std::move(member_inputs)));
  return *insert(func->name(), methods_, EntityType::METHOD, std::move(m));
}

void Module::lift_lowered_methods(size_t start) {
  for(size_t i = start; i < lowered_methods_.get_functions().size(); ++i) {
    Method& m = _create_lowered_method(
        lowered_methods_.get_functions().at(i).get(), {});
    lift_lowered_method(m);
  }
}

void Module::_define_lowered(
    const std::vector<Def>& definitions,
    const std::vector<Resolver>& resolvers) {
  size_t start = lowered_methods_.get_functions().size();
  lowered_methods_.define(definitions, resolvers, nullptr);
  lift_lowered_methods(start);
  // call lift_lowered_method for each definition
}

void Module::_define_lowered(const std::string& src, const Resolver& resolver) {
  size_t start = lowered_methods_.get_functions().size();
  lowered_methods_.define(src, resolver, nullptr);
  lift_lowered_methods(start);
}

Method& Module::_define_lowered(std::string name, std::shared_ptr<Graph> graph, std::vector<Slot> slots) {
  Method& m = _create_lowered_method(&lowered_methods_.create_function(std::move(name), std::move(graph)), std::move(slots));
  lift_lowered_method(m);
  return m;
}

} // namespace script
} // namespace jit
} // namespace torch
