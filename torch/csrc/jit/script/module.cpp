#include <c10/util/Exception.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/schema_matching.h>

namespace torch {
namespace jit {
namespace script {

// first class mode runs models as first class objects,
// and does not force inlining everywhere. This is experimental
// as we bring up the system since it will degrade performance
// and may introduce bugs. test_jit.py provides context managers
// that enable it for specific tests.
thread_local bool inline_everything = true;
bool& getInlineEverythingMode() {
  return inline_everything;
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

void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) const {
  ExportModule(*this, out, extra_files);
}

void Module::save(const std::string& filename, const ExtraFilesMap& extra_files)
    const {
  ExportModule(*this, filename, extra_files);
}

void module_state_to(
    const Slot& s,
    const c10::optional<at::Device>& device,
    const c10::optional<at::ScalarType>& dtype,
    bool non_blocking) {
  // Need to access the `at::Tensor` as a `Variable` here.
  autograd::Variable variable = s.value().toTensor();
  // Use the data's original device or dtype if not supplied here.
  auto new_data = variable.to(
      device.value_or(variable.device()),
      dtype.value_or(variable.scalar_type()),
      non_blocking);
  variable.set_data(new_data);
}

void Module::to_impl(
    const c10::optional<at::Device>& device,
    const c10::optional<at::ScalarType>& dtype,
    bool non_blocking) {
  // First call `to()` on every child module.
  for (Module child : get_modules()) {
    child.to_impl(device, dtype, non_blocking);
  }
  // Then convert every of our parameters.
  for (Slot parameter : get_parameters()) {
    module_state_to(parameter, device, dtype, non_blocking);
  }
  // Then convert every tensor attributes (buffers).
  for (Slot attr : get_attributes()) {
    if (attr.type()->isSubtypeOf(TensorType::get())) {
      module_state_to(attr, device, dtype, non_blocking);
    }
  }
}

// remove the first module argument, replacing any access of its
// parameters/attributes with extra_ivalue input Slots that hold what value to
// pass into the graph. Used for ONNX export to remove first-class modules
// so it can deal purely with parameters and inputs
std::pair<std::shared_ptr<Graph>, std::vector<Slot>> lower_graph(
    const ModulePtr& self,
    Graph& g_,
    size_t self_offset = 0) {
  std::shared_ptr<Graph> g = g_.copy();
  std::vector<Slot> extra_ivalues;
  std::unordered_map<Slot, size_t> slot_to_offset;
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
      for (const Slot& slot : new_slots) {
        e.n->addInput(getOrAddSlot(slot));
      }
      e.n->removeInput(e.offset);
      continue;
    }
    if (e.n->kind() != prim::GetAttr) {
      throw ErrorReport(e.n->sourceRange())
          << "temporary: the only valid use of a module is looking up an "
             "attribute but found "
          << *e.n;
    }
    Slot slot(e.mod, e.mod->type()->getAttributeSlot(e.n->s(attr::name)));
    if (ClassTypePtr c = e.n->output()->type()->cast<ClassType>()) {
      if (c->is_module()) {
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

  return std::make_pair(std::move(g), std::move(extra_ivalues));
}

Method::Method(ModulePtr owner, std::shared_ptr<Function> function)
    : owner_(std::move(owner)), function_(std::move(function)) {}

Module Method::owner() const {
  return Module(owner_);
}
void Method::run(Stack& stack) {
  stack.insert(stack.begin(), owner().module_object());
  function_->run(stack);
}

IValue Method::operator()(std::vector<IValue> stack, const Kwargs& kwargs) {
  stack.insert(stack.begin(), owner().module_object());
  return (*function_)(std::move(stack), kwargs);
}

static std::vector<at::Tensor> loadTensors(const std::vector<Slot>& slots) {
  std::vector<at::Tensor> result;
  result.reserve(slots.size());
  for(const Slot& slot : slots) {
    result.emplace_back(slot.value().toTensor());
  }
  return result;
}
std::pair<std::shared_ptr<Graph>, std::vector<at::Tensor>> Method::_lowered_graph() {
  auto result = lower_graph(owner().module_object(), *graph());
  return std::make_pair(result.first, loadTensors(result.second));
}

void Module::define(const std::string& src, const ResolverPtr& resolver) {
  class_compilation_unit()->define(
      src,
      resolver ? resolver : script::nativeResolver(),
      simpleSelf(module_object()->type()));
}

void Module::copy_into(
    const ModuleLookup& module_lookup,
    // translate current module singleton type to new module
    // singleton type.
    std::unordered_map<TypePtr, TypePtr>& type_remap,
    std::vector<std::string> names) const {
  auto curr = module_lookup(names);
  type_remap[module_object()->type()] = curr.module_object()->type();

  for (Slot s : curr.get_slots()) {
    if (s.is_module()) {
      names.push_back(s.name());
      // Submodules must be translated first, otherwise parameter_remap entries
      // will not be filled in for methods of this module.
      s.to_module().copy_into(module_lookup, type_remap, names);
      names.pop_back();
    } else {
      curr.set_or_add_slot(s.name(), s.type(), s.value(), s.entity_type());
    }
  }

  for (auto& fn : class_compilation_unit()->get_functions()) {
    curr.clone_method(*this, fn->name(), type_remap);
  }
}

void Module::clone_method(
    const Module& orig,
    const std::string& name,
    const std::unordered_map<TypePtr, TypePtr>& type_remap) {
  // type remapping - when we copy method implementations from one module
  // singleton to another, we need to update the types of the self arguments
  // to match the new module.
  // XXX - this only handles modules that occur as variables, not modules
  // that appear in aggregate types. Currently this works fine because
  // we restrict how modules can be used during the lowering step. Eventually,
  // we will need to decide what it means for us to 'copy' a module.
  // For instance, we can copy just the state (parameters, attributes),
  // but share the code. Or we can copy the code. If we choose to copy the
  // code, what should we do about aggregate types that contain a module?
  auto type_remap_fn = [&](TypePtr in) {
    auto it = type_remap.find(in);
    if (it == type_remap.end())
      return in;
    return it->second;
  };
  const Function& fn = orig.class_compilation_unit()->get_function(name);
  auto graph = fn.graph()->copy();
  graph->remapTypes(type_remap_fn);
  auto schema = fn.getSchema().cloneWithRemappedTypes(type_remap_fn);
  auto copied = class_compilation_unit()->create_function(fn.name(), graph);
  copied->setSchema(std::move(schema));
}

void Module::clone_method(const Module& orig, const std::string& name) {
  std::unordered_map<TypePtr, TypePtr> type_remap;
  std::vector<std::pair<Module, Module>> to_scan = {{orig, *this}};
  while (!to_scan.empty()) {
    auto entry = to_scan.back();
    to_scan.pop_back();
    type_remap[entry.first.module_object()->type()] =
        entry.second.module_object()->type();
    for (Slot s : entry.first.get_module_slots()) {
      to_scan.emplace_back(s.to_module(), entry.second.get_module(s.name()));
    }
  }
  return clone_method(orig, name, type_remap);
}

void Module::train(bool on) {
  for (auto submod : get_modules()) {
    submod.train(on);
  }
  if (auto slot = find_attribute("training")) {
    slot->setValue(on);
  } else {
    register_attribute("training", BoolType::get(), on);
  }
}

IValue Module::create_class(const c10::QualifiedName& name, Stack stack) const {
  // Look up the class
  const auto classType =
      class_compilation_unit()->get_class(c10::QualifiedName(name));
  if (!classType) {
    AT_ERROR(
        "Could not find class with name: '",
        name.qualifiedName(),
        "' in module.");
  }

  // Create a bare object with correct number of slots
  const size_t numAttrs = classType->numAttributes();
  auto obj = c10::ivalue::Object::create(classType, numAttrs);

  // Invoke the `__init__()` of the class with the arguments provided.
  Stack stackWithSelf = {obj};
  for (auto& arg : stack) {
    stackWithSelf.push_back(std::move(arg));
  }
  // Note: following Python, `__init__()` modifies its first parameter in-place
  // and returns nothing.
  classType->getMethod("__init__")->operator()(std::move(stackWithSelf));

  return obj;
}

slot_list Module::get_parameters() const {
  return slot_list(*this, EntityType::PARAMETER);
}

slot_list Module::get_attributes() const {
  return slot_list(*this, EntityType::ATTRIBUTE);
}

slot_list Module::get_module_slots() const {
  return slot_list(*this, EntityType::MODULE);
}

slot_list Module::get_slots() const {
  return slot_list(*this, c10::nullopt);
}

Module Slot::to_module() const {
  return Module(value().toObject());
}

module_list Module::get_modules() const {
  return module_list(*this, EntityType::MODULE);
}

void Module::apply(const std::function<void(Module&)>& fn) {
  for (auto submod : get_modules()) {
    submod.apply(fn);
  }
  fn(*this);
}

} // namespace script
} // namespace jit
} // namespace torch
