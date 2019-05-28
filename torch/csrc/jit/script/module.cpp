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
        << " method '" << name() << "' is called recursively. "
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
  for (auto& child : get_modules()) {
    child->to_impl(device, dtype, non_blocking);
  }
  // Then convert every of our parameters.
  for (auto& parameter : get_parameters()) {
    module_state_to(parameter, device, dtype, non_blocking);
  }
  // Then convert every tensor attributes (buffers).
  for (auto& attr : get_attributes()) {
    if (attr.type()->isSubtypeOf(TensorType::get())) {
      module_state_to(attr, device, dtype, non_blocking);
    }
  }
}

// lower_first_class_method and lift_lowered_method are transitionary functions
// used to translate between module-as-first-class code generation,
// and module-as-special execution. Once module-as-first-class execution is
// debugged, then we can remove both and remove the lowered_functions_ table.

// remove the first module argument, replacing any access of its
// parameters/attributes with extra_ivalue input Slots that hold what value to
// pass into the graph
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
      if (c->qualname() == "__torch__.$Module") {
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

std::pair<std::shared_ptr<Function>, std::vector<Slot>> Module::
    lower_first_class_method(Function* fn) {
  fn->ensure_defined();
  auto lowered = lower_graph(module_object(), *fn->graph());
  CompilationUnit cu;
  cu.set_optimized(fn->is_optimized());
  std::shared_ptr<Function> new_func =
      cu.create_function(fn->name(), lowered.first);

  // generate the new schema
  // slice away the self argument
  std::vector<Argument> args(
      fn->getSchema().arguments().begin() + 1,
      fn->getSchema().arguments().end());
  size_t id = 0;
  for (const Slot& slot : lowered.second) {
    std::ostringstream ss;
    ss << "slot" << id++;
    args.emplace_back(ss.str(), slot.type());
  }
  new_func->setSchema(fn->getSchema().cloneWithArguments(std::move(args)));
  return std::make_pair(new_func, std::move(lowered.second));
}

static FunctionSchema sliceFirst(const FunctionSchema& schema) {
  // we are required to slice out the self argument
  // because it is not expected to appear in Module schema
  // until the executor is made to be first-class
  std::vector<Argument> sliced(
      schema.arguments().begin() + 1, schema.arguments().end());
  return schema.cloneWithArguments(std::move(sliced));
}

Method::Method(Module* owner, Function* first_class_function)
    : owner_(owner), schema_(sliceFirst(first_class_function->getSchema())) {
  std::tie(function_, initial_ivalues_) =
      owner->lower_first_class_method(first_class_function);
}

void Module::define(const std::string& src, const ResolverPtr& resolver) {
  class_compilation_unit().define(
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
  type_remap[module_object()->type()] = curr->module_object()->type();
  for (auto& param : get_parameters()) {
    curr->register_parameter(
        param.name(),
        param.value().toTensor(),
        /*is_buffer=*/false);
  }
  for (auto& attr : get_attributes()) {
    curr->register_attribute(attr.name(), attr.type(), attr.value());
  }

  for (auto& mod : get_modules()) {
    names.push_back(mod->name());
    // Submodules must be translated first, otherwise parameter_remap entries
    // will not be filled in for methods of this module.
    mod->copy_into(module_lookup, type_remap, names);
    names.pop_back();
  }

  for (auto& fn : class_compilation_unit().get_functions()) {
    curr->clone_method(*this, fn->name(), type_remap);
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
  const Function& fn = orig.class_compilation_unit().get_function(name);
  auto graph = fn.graph()->copy();
  graph->remapTypes(type_remap_fn);
  auto schema = fn.getSchema().cloneWithRemappedTypes(type_remap_fn);
  auto copied = class_compilation_unit().create_function(fn.name(), graph);
  copied->setSchema(std::move(schema));
}

void Module::clone_method(const Module& orig, const std::string& name) {
  std::unordered_map<TypePtr, TypePtr> type_remap;
  std::vector<std::pair<const Module*, const Module*>> to_scan = {
      {&orig, this}};
  while (!to_scan.empty()) {
    auto entry = to_scan.back();
    to_scan.pop_back();
    type_remap[entry.first->module_object()->type()] =
        entry.second->module_object()->type();
    for (const auto& sub : entry.first->get_modules()) {
      to_scan.emplace_back(
          sub.get(), entry.second->get_module(sub->name()).get());
    }
  }
  return clone_method(orig, name, type_remap);
}

void Module::train(bool on) {
  for (auto& submod : get_modules()) {
    submod->train(on);
  }
  register_buffer("training", torch::tensor(on ? 1 : 0, at::kLong));
}

IValue Module::create_class(const c10::QualifiedName& name, Stack stack) const {
  // Classes live in the top-level compilation unit.
  if (parent_) {
    return parent_->create_class(name, std::move(stack));
  }

  // Look up the class
  const auto classType =
      class_compilation_unit().get_class(c10::QualifiedName(name));
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

} // namespace script
} // namespace jit
} // namespace torch
