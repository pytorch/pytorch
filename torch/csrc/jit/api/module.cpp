#include <ATen/core/symbol.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch::jit {

namespace {

std::string getInputDebugName(const Node& n, const int idx) {
  return n.inputs().at(idx)->debugName();
}

void assert_ignored_methods_not_called(
    torch::jit::Function& fn,
    const std::unordered_set<std::string>& ignored_methods) {
  if (ignored_methods.empty()) {
    return;
  }
  const bool recurse = true;
  std::vector<Node*> all_nodes = findAllNodes(
      *toGraphFunction(fn).graph(), c10::prim::CallMethod, recurse);

  // Extract method names from these nodes.
  std::unordered_set<std::string> encountered_ignored_methods;

  for (Node* n : all_nodes) {
    if (ignored_methods.count(n->s(attr::name)) > 0 &&
        getInputDebugName(*n, 0) == "self") {
      encountered_ignored_methods.insert(
          getInputDebugName(*n, 0) + "." + n->s(attr::name));
    }
  }
  if (encountered_ignored_methods.empty()) {
    return;
  }

  const std::string encountered_ignored_methods_str =
      c10::Join(", ", encountered_ignored_methods);

  TORCH_CHECK(
      false,
      "Preserved method '",
      fn.name(),
      "' references ignored method(s) '",
      encountered_ignored_methods_str,
      "'. This is not permitted.");
}

void assert_ignored_attributes_not_referenced(
    torch::jit::Function& fn,
    const std::unordered_set<std::string>& ignored_attributes) {
  if (ignored_attributes.empty()) {
    return;
  }

  const bool recurse = true;
  std::vector<Node*> all_nodes =
      findAllNodes(*toGraphFunction(fn).graph(), c10::prim::GetAttr, recurse);

  // Extract attribute names from these nodes.
  std::unordered_set<std::string> encountered_ignored_attributes;

  for (Node* n : all_nodes) {
    if (ignored_attributes.count(n->s(attr::name)) > 0 &&
        getInputDebugName(*n, 0) == "self") {
      encountered_ignored_attributes.insert(
          getInputDebugName(*n, 0) + "." + n->s(attr::name));
    }
  }
  if (encountered_ignored_attributes.empty()) {
    return;
  }

  const std::string encountered_ignored_attributes_str =
      c10::Join(", ", encountered_ignored_attributes);

  TORCH_CHECK(
      false,
      "Preserved method '",
      fn.name(),
      "' references ignored attribute(s) '",
      encountered_ignored_attributes_str,
      "'. This is not permitted.");
}

} // namespace

static ObjectPtr create_module_object(
    c10::QualifiedName class_name,
    std::shared_ptr<CompilationUnit> cu,
    bool shouldMangle = false) {
  // If the name is unqualified, prepend a `__torch__`, similar to what Python
  // does with `__main__` for top-level code.
  if (class_name.prefix().empty()) {
    class_name = c10::QualifiedName("__torch__", class_name.name());
  }
  if (shouldMangle && cu->get_class(class_name) != nullptr) {
    class_name = cu->mangle(class_name);
  }
  auto cls = ClassType::create(std::move(class_name), cu, /*is_module=*/true);
  cu->register_type(cls);
  return c10::ivalue::Object::create(
      c10::StrongTypePtr(std::move(cu), std::move(cls)), 0);
}

Module::Module(c10::QualifiedName class_name)
    : Object(create_module_object(
          std::move(class_name),
          std::make_shared<CompilationUnit>())) {}

Module::Module(
    std::shared_ptr<CompilationUnit> cu,
    const c10::ClassTypePtr& type)
    : Object(c10::ivalue::Object::create(
          c10::StrongTypePtr(std::move(cu), type),
          type->numAttributes())) {}

Module::Module(
    c10::QualifiedName class_name,
    std::shared_ptr<CompilationUnit> cu,
    bool shouldMangle)
    : Object(create_module_object(
          std::move(class_name),
          std::move(cu),
          shouldMangle)) {}

// first class mode runs models as first class objects,
// and does not force inlining everywhere. This is experimental
// as we bring up the system since it will degrade performance
// and may introduce bugs. test_jit.py provides context managers
// that enable it for specific tests.
thread_local bool inline_everything = false;
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

void module_state_to(
    const autograd::Variable& variable,
    const c10::optional<at::Device>& device,
    const c10::optional<at::ScalarType>& dtype,
    bool non_blocking) {
  // Need to access the `at::Tensor` as a `Variable` here.
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
  for (at::Tensor e : parameters()) {
    module_state_to(e, device, dtype, non_blocking);
  }
  for (at::Tensor e : buffers()) {
    module_state_to(e, device, dtype, non_blocking);
  }
}

Method::Method(ModulePtr owner, Function* function)
    : owner_(std::move(owner)), function_(function) {}

Module Method::owner() const {
  return Module(owner_);
}
void Method::run(Stack& stack) {
  stack.insert(stack.begin(), owner()._ivalue()); // self
  RECORD_TORCHSCRIPT_FUNCTION(name(), stack);
  function_->run(stack);
}

IValue Method::operator()(std::vector<IValue> stack, const Kwargs& kwargs)
    const {
  stack.insert(stack.begin(), owner()._ivalue()); // self
  RECORD_TORCHSCRIPT_FUNCTION(name(), stack);
  return (*function_)(std::move(stack), kwargs);
}

c10::intrusive_ptr<c10::ivalue::Future> Method::run_async(
    std::vector<IValue> stack,
    const Kwargs& kwargs,
    TaskLauncher taskLauncher) {
  stack.insert(stack.begin(), owner()._ivalue());
  RECORD_TORCHSCRIPT_FUNCTION(name(), stack);

  function_->getSchema().checkAndNormalizeInputs(stack, kwargs);
  return function_->runAsync(stack, std::move(taskLauncher));
}

void Method::setArgumentNames(
    std::vector<std::string>& argumentNamesOut) const {
  TORCH_INTERNAL_ASSERT(function_);
  auto& arguments = function_->getSchema().arguments();
  argumentNamesOut.reserve(arguments.size());
  for (auto& argument : arguments) {
    if (argument.name() == "self") {
      continue;
    }
    argumentNamesOut.push_back(argument.name());
  }
}

IValue Module::operator()(std::vector<IValue> inputs) {
  const auto& pre_forward_hooks = type()->getForwardPreHooks();
  const auto& forward_hooks = type()->getForwardHooks();

  // call forward pre_hooks
  for (const auto& pre_hook : pre_forward_hooks) {
    auto tuple_input = c10::ivalue::Tuple::create(inputs);
    IValue result = Method(_ivalue(), pre_hook)({tuple_input});
    if (!result.isNone()) {
      if (result.isTuple()) {
        inputs = result.toTupleRef().elements().vec();
      } else {
        inputs = {result};
      }
    }
  }

  // call forward
  auto outputs = forward(inputs);

  // call forward hooks
  for (const auto& hook : forward_hooks) {
    auto tuple_input = c10::ivalue::Tuple::create(inputs);
    auto hook_result = Method(_ivalue(), hook)({tuple_input, outputs});
    if (!hook_result.isNone()) {
      outputs = hook_result;
    }
  }
  return outputs;
}

void Module::clone_method(
    const Module& orig,
    const Function& method,
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
  auto graph = toGraphFunction(method).graph()->copy();
  graph->remapTypes(type_remap_fn);
  auto schema = method.getSchema().cloneWithRemappedTypes(type_remap_fn);
  const auto this_method_name = getNameForMethod(method.name());
  auto copied =
      _ivalue()->compilation_unit()->create_function(this_method_name, graph);
  type()->addMethod(copied);
  copied->setSchema(std::move(schema));
}

void Module::clone_method(const Module& orig, const std::string& name) {
  std::unordered_map<TypePtr, TypePtr> type_remap;
  std::vector<std::pair<Module, Module>> to_scan = {{orig, *this}};
  while (!to_scan.empty()) {
    auto entry = to_scan.back();
    to_scan.pop_back();
    type_remap[entry.first._ivalue()->type()] = entry.second._ivalue()->type();
    for (const NameModule& s : entry.first.named_children()) {
      to_scan.emplace_back(
          s.value, Module(entry.second.attr(s.name).toObject()));
    }
  }
  return clone_method(orig, orig.get_method(name).function(), type_remap);
}

Module Module::copy() const {
  return Module(_ivalue()->copy());
}

Module Module::deepcopy() const {
  return Module(_ivalue()->deepcopy());
}

Module Module::clone(bool inplace) const {
  std::unordered_map<TypePtr, TypePtr> type_remap;
  IValue::HashAliasedIValueMap memo;
  const std::unordered_set<std::string> ignored_methods;
  const std::unordered_set<std::string> ignored_attributes;
  return clone_impl(
      type_remap, inplace, memo, ignored_methods, ignored_attributes);
}

Module Module::clone(
    bool inplace,
    const std::unordered_set<std::string>& ignored_methods,
    const std::unordered_set<std::string>& ignored_attributes) const {
  std::unordered_map<TypePtr, TypePtr> type_remap;
  IValue::HashAliasedIValueMap memo;
  return clone_impl(
      type_remap, inplace, memo, ignored_methods, ignored_attributes);
}

Module Module::clone_impl(
    std::unordered_map<TypePtr, TypePtr>& type_remap,
    bool inplace,
    IValue::HashAliasedIValueMap memo,
    const std::unordered_set<std::string>& ignored_methods,
    const std::unordered_set<std::string>& ignored_attributes) const {
  // Create a new _ivalue in the same compilation unit.
  // Since now we have shared ClassType, we need to preserve the shared
  // ClassType during cloning, so we first need to check if the type
  // is already cloned, if so, we'll create a new module with the cloned
  // ClassType, if not, we'll create a new module and a new ClassType.
  bool type_already_cloned = type_remap.find(type()) != type_remap.end();
  Module r;
  if (type_already_cloned) {
    // if we cloned the class type before, we'll reuse it
    Module new_module(
        _ivalue()->compilation_unit(), type_remap[type()]->cast<ClassType>());
    r = new_module;
  } else {
    Module new_module(*type()->name(), _ivalue()->compilation_unit(), true);
    r = new_module;
    type_remap[type()] = r.type();
  }

  // Copy slots. If a slot is a module - recursively clone it.
  size_t N = type()->numAttributes();
  for (const auto i : c10::irange(N)) {
    IValue s = _ivalue()->getSlot(i);
    std::string attr_name = type()->getAttributeName(i);

    // If this attribute is in the list of ignored attributes, skip it
    // (i.e. do not clone it).
    if (ignored_attributes.count(attr_name) != 0) {
      continue;
    }

    TypePtr attr_type = type()->getAttribute(i);
    if (attr_type->is_module()) {
      const Module& orig = Module(s.toObject());
      const std::unordered_set<std::string> empty_set;
      Module cloned =
          orig.clone_impl(type_remap, inplace, memo, empty_set, empty_set);
      type_remap[orig.type()] = cloned.type();
      // NOTE: why do we need to manually setattr on object instead of using
      // register_module here? because the attr can be a module interface
      // type and hold a Module object still. register_module will not let us
      // correctly set up the type for this attr, so we had to do this manually.
      // In the case it's an interface type, the type will be shared by the new
      // cloned instance in the same compilation unit bc it only contains a list
      // of functionSchema
      r.type()->addOrCheckAttribute(
          attr_name, attr_type->cast<ClassType>() ? cloned.type() : attr_type);
      r._ivalue()->setAttr(attr_name, cloned._ivalue());
    } else {
      // this adds new slot and creates a new attribute for the underlying type
      // if the type is not already cloned, otherwise it will only add a new
      // slot and typecheck
      r.register_attribute(
          type()->getAttributeName(i),
          attr_type,
          // we'll deepcopy the IValue in non inplace option
          inplace ? s : s.deepcopy(memo),
          type()->is_parameter(i),
          type()->is_buffer(i));
    }
  }

  // only clone the methods if the ClassType is not cloned before
  if (!type_already_cloned) {
    // clone constants
    for (size_t i = 0; i < type()->numConstants(); ++i) {
      r.type()->addConstant(type()->getConstantName(i), type()->getConstant(i));
    }
    // clone methods, remapping the types to the cloned ones.
    for (auto& fn : type()->methods()) {
      // If this method is not in the list of ignored methods, clone it.
      if (ignored_methods.count(fn->name()) == 0) {
        assert_ignored_methods_not_called(*fn, ignored_methods);
        assert_ignored_attributes_not_referenced(*fn, ignored_attributes);
        r.clone_method(*this, *fn, type_remap);
      }
    }

    // Execute __setstate__(__getstate__()) to initialize custom class members.
    if (auto setstate_method = r.find_method("__setstate__")) {
      auto getstate_method = r.find_method("__getstate__");
      TORCH_INTERNAL_ASSERT(getstate_method, "expect __getstate__");
      auto state = (*getstate_method)(Stack{});
      (*setstate_method)(Stack{state});
    }
  }
  return r;
}

void Module::train(bool on) {
  for (Module m : modules()) {
    if (auto slot = m._ivalue()->type()->findAttributeSlot("training")) {
      m._ivalue()->setSlot(*slot, on);
    } else {
      // FIXME[T110620981]: This assert was broken (never asserted), and once
      // fixed it triggers test failures.  Fix me!
      /* TORCH_INTERNAL_ASSERT(false, "'training' attribute not found"); */
    }
  }
}

IValue Module::create_class(const c10::QualifiedName& name, Stack stack) const {
  // Look up the class
  const auto classType =
      _ivalue()->compilation_unit()->get_class(c10::QualifiedName(name));
  if (!classType) {
    AT_ERROR(
        "Could not find class with name: '",
        name.qualifiedName(),
        "' in module.");
  }

  // Create a bare object with correct number of slots
  const size_t numAttrs = classType->numAttributes();
  auto obj = c10::ivalue::Object::create(
      c10::StrongTypePtr(_ivalue()->compilation_unit(), classType), numAttrs);

  // Invoke the `__init__()` of the class with the arguments provided.
  Stack stackWithSelf = {obj};
  for (auto& arg : stack) {
    stackWithSelf.push_back(std::move(arg));
  }
  // Note: following Python, `__init__()` modifies its first parameter in-place
  // and returns nothing.
  classType->getMethod("__init__").operator()(std::move(stackWithSelf));

  return obj;
}

Module freeze(
    const Module& module,
    c10::optional<std::vector<std::string>> preserved_attrs,
    bool optimize_numerics) {
  TORCH_CHECK(
      !module.hasattr("training") || !module.is_training(),
      "Freezing is currently only implemented for modules in eval mode. Please call .eval() before freezing");

  Module out_mod = freeze_module(
      module, preserved_attrs.value_or(std::vector<std::string>({})));
  auto graph = out_mod.get_method("forward").graph();
  OptimizeFrozenGraph(graph, optimize_numerics);
  return out_mod;
}

namespace {
void optimize_for_inference(std::shared_ptr<Graph> graph) {
  FuseFrozenConvAddRelu(graph);
  ConvertFrozenOpsToMKLDNN(graph);
  FrozenLinearTranspose(graph);
}
} // namespace

Module optimize_for_inference(
    Module& module,
    const std::vector<std::string>& other_methods) {
  // if not frozen yet
  Module frozen_mod;
  if (module._ivalue()->type()->hasAttribute("training")) {
    frozen_mod = freeze(module, {}, true);
  } else {
    frozen_mod = module;
  }

  optimize_for_inference(frozen_mod.get_method("forward").graph());

  for (const auto& method : other_methods) {
    optimize_for_inference(frozen_mod.get_method(method).graph());
  }
  return frozen_mod;
}

buffer_list Module::buffers(bool recurse) const {
  return buffer_list(*this, recurse, /*return_module=*/false);
}
named_buffer_list Module::named_buffers(bool recurse) const {
  return named_buffer_list(*this, recurse, /*return_module=*/false);
}

module_list Module::children() const {
  return module_list(*this, /*recurse=*/false, /*return_module=*/false);
}
named_module_list Module::named_children() const {
  return named_module_list(*this, /*recurse=*/false, /*return_module=*/false);
}
module_list Module::modules() const {
  return module_list(*this, /*recurse=*/true, /*return_module=*/true);
}
named_module_list Module::named_modules() const {
  return named_module_list(*this, /*recurse=*/true, /*return_module=*/true);
}

parameter_list Module::parameters(bool recurse) const {
  return parameter_list(*this, recurse, /*return_module=*/false);
}
named_parameter_list Module::named_parameters(bool recurse) const {
  return named_parameter_list(*this, recurse, /*return_module=*/false);
}

attribute_list Module::attributes(bool recurse) const {
  return attribute_list(*this, recurse, /*return_module=*/false);
}
named_attribute_list Module::named_attributes(bool recurse) const {
  return named_attribute_list(*this, recurse, /*return_module=*/false);
}

void Module::apply(const std::function<void(Module&)>& fn) {
  for (Module s : modules()) {
    fn(s);
  }
}

std::string Module::dump_to_str(
    bool print_method_bodies,
    bool print_attr_values,
    bool print_param_values) const {
  std::stringstream ss;
  std::stringstream parameters_ss;
  std::stringstream attributes_ss;
  std::stringstream methods_ss;
  std::stringstream submodules_ss;

  for (const NameTensor& p : named_parameters(/*recurse=*/false)) {
    parameters_ss << p.name << " = ";
    if (print_param_values) {
      parameters_ss << p.value << std::endl;
    } else {
      parameters_ss << "..." << std::endl;
    }
  }

  for (const NameValue& p : named_attributes(/*recurse=*/false)) {
    attributes_ss << p.name << " = ";
    if (!p.value.isTensor() || print_attr_values) {
      attributes_ss << p.value << std::endl;
    } else {
      attributes_ss << "..." << std::endl;
    }
  }

  for (const Method& method : get_methods()) {
    methods_ss << "  method " << method.name() << " {" << std::endl;
    if (print_method_bodies) {
      methods_ss << torch::jit::jit_log_prefix(
                        "    ", method.graph()->toString())
                 << std::endl;
    }
    methods_ss << "  }" << std::endl;
  }

  ss << "module " << type()->name()->qualifiedName() << " {" << std::endl;
  ss << "  parameters {" << std::endl;
  ss << torch::jit::jit_log_prefix("    ", parameters_ss.str());
  ss << "  }" << std::endl;
  ss << "  attributes {" << std::endl;
  ss << torch::jit::jit_log_prefix("    ", attributes_ss.str());
  ss << "  }" << std::endl;
  ss << "  methods {" << std::endl;
  ss << torch::jit::jit_log_prefix("  ", methods_ss.str());
  ss << "  }" << std::endl;
  ss << "  submodules {" << std::endl;
  for (const NameModule& s : named_children()) {
    // We do 4 spaces here, because one level of indentation comes from
    // 'submodules' scope and the other one goes from a specific submodule we're
    // printing.
    ss << torch::jit::jit_log_prefix(
        "    ",
        s.value.dump_to_str(
            print_method_bodies, print_attr_values, print_param_values));
  }
  ss << "  }" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

void Module::dump(
    bool print_method_bodies = true,
    bool print_attr_values = true,
    bool print_param_values = true) const {
  std::cout << dump_to_str(
                   print_method_bodies, print_attr_values, print_param_values)
            << std::endl;
}

} // namespace torch::jit

namespace c10 {

torch::jit::Module IValue::toModule() const {
  return torch::jit::Module(toObject());
}
bool IValue::isModule() const {
  return isObject() && toObjectRef().type()->is_module();
}

} // namespace c10
