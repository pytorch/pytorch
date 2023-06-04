#include <ATen/core/class_type.h>

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include <ATen/core/grad_mode.h>
#include <ATen/core/function.h>

namespace c10 {

void ClassType::addMethod(torch::jit::Function* method) {
  TORCH_CHECK(
      findMethod(method->name()) == nullptr,
      "Can't redefine method: ",
      method->name(),
      " on class: ",
      repr_str());
  methods_.push_back(method);
}

const std::vector<torch::jit::Function*>& ClassType::getForwardHooks() const {
    return forward_hooks_;
}

const std::vector<torch::jit::Function*>& ClassType::getForwardPreHooks() const {
    return forward_pre_hooks_;
}

void ClassType::addForwardPreHook(torch::jit::Function* pre_hook_ptr) {
    forward_pre_hooks_.emplace_back(pre_hook_ptr);
}

void ClassType::addForwardHook(torch::jit::Function* hook_ptr) {
    forward_hooks_.emplace_back(hook_ptr);
}

torch::jit::Function* ClassType::findForwardPreHook(const std::string& name) const {
  for (const auto& pre_hook : forward_pre_hooks_) {
    if (name == pre_hook->name()) {
      return pre_hook;
    }
  }
  return nullptr;
}

torch::jit::Function* ClassType::findForwardHook(const std::string& name) const {
  for (const auto& hook : forward_hooks_) {
    if (name == hook->name()) {
      return hook;
    }
  }
  return nullptr;
}

static std::string getSchemaInputTypesString(const FunctionSchema& schema) {
  std::stringstream input_types;
  const std::vector<Argument>& forward_args = schema.arguments();
  for (const auto i : c10::irange(1, forward_args.size())) {
    input_types << forward_args[i].type()->annotation_str();
    if (forward_args.size() - 1 != i) {
      input_types << ", ";
    }
  }
  if (forward_args.size() == 1) {
    input_types << "()";
  }
  return input_types.str();
}

std::string ClassType::getForwardPreHookErrorMessage(int pre_hook_idx) const {
  const std::string& pre_hook_name = forward_pre_hooks_[pre_hook_idx]->name();
  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  std::string input_types = getSchemaInputTypesString(forward_schema);
  const std::vector<Argument>& forward_args = forward_schema.arguments();

  std::string single_output = "";
  if (forward_args.size() == 2 &&
      forward_args[1].type()->cast<TupleType>() == nullptr) {
    // if the output type is a single tuple, it needs to be wrapped in an outer tuple
    // to match eager's behavior
    single_output = ", '" + forward_args[1].type()->annotation_str() + "',";
  }
  std::string pre_hook_schema =
      pre_hook_name + "(self, input: Tuple[" + input_types + "])";
  std::string return_string =
      "This error occurred while scripting the forward pre-hook '" +
      pre_hook_name + "' on module '" + name()->name() +
      "'. If you did not want to script this pre-hook remove it from the "
      "original NN module before scripting. Pre-hooks for module '" +
      name()->name() + "' are expected to have the following signature: "
      + pre_hook_schema + " with a return type of either 'None'" +
      single_output + " or 'Tuple[" + input_types + "]'.";
  return return_string;
}

std::string ClassType::getForwardHookErrorMessage(int hook_idx) const {
  const std::string& hook_name = forward_hooks_[hook_idx]->name();
  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  std::string input_types = getSchemaInputTypesString(forward_schema);

  // create expected output types string
  const Argument& pre_output =
      (hook_idx == 0)
          ? forward_schema.returns()[0]
          : forward_hooks_[hook_idx - 1]->getSchema().returns()[0];
  std::string output_types = pre_output.type()->annotation_str();
  // create error message
  std::string hook_schema = hook_name + "(self, input: Tuple[" +
                            input_types + "], output: " + output_types + ")";
  std::string return_string =
      "This error occurred while scripting the forward hook '"
      + hook_name + "' on module " + name()->name() +
      ". If you did not want to script this hook remove it from" +
      " the original NN module before scripting. This hook was" +
      " expected to have the following signature: " + hook_schema +
      ". The type of the output arg is the returned type from" +
      " either the forward method or the previous hook if it exists. " +
      "Note that hooks can return anything, but if the hook is " +
      "on a submodule the outer module is expecting" +
      " the same return type as the submodule's forward.";
  return return_string;
}

bool ClassType::isUnresolvedClassAttribute(const std::string& name) const {
  return std::find(
      unresolved_class_attributes_.begin(),
      unresolved_class_attributes_.end(),
      name) != unresolved_class_attributes_.end();
}

static void checkForwardHookInputArguments(
    const FunctionSchema& forward_schema,
    const FunctionSchema& hook_schema,
    const std::string& hook_id,
    const std::string& hook_err_msg) {
  // check for proper tuple input types
  const std::vector<Argument>& forward_args = forward_schema.arguments();
  const Argument input_arg = hook_schema.arguments()[1];
  TORCH_CHECK(
      input_arg.type()->cast<TupleType>() != nullptr,
      hook_id,
      "expected the input argument to be typed as a Tuple but found type: '",
      input_arg.type()->annotation_str(),
      "' instead.\n",
      hook_err_msg
   );

  const at::ArrayRef<TypePtr> input_tuple_types = input_arg.type()->castRaw<TupleType>()->elements();
  if (forward_args.size() == 1) {
    // check for empty forward case
    TORCH_CHECK(
        input_tuple_types.empty(),
        hook_id,
        "was expecting Tuple[()] as the input type. Received type: '",
        input_arg.type()->annotation_str(),
        "'.\n",
        hook_err_msg
      );
  } else {
    // check input tuple for correct size and correct contained types
    TORCH_CHECK(
        input_tuple_types.size() == forward_args.size() - 1,
        hook_id,
        "has the wrong number of contained types for the",
        " input argument's Tuple. Received type: '",
        input_arg.type()->annotation_str(),
        "'.\n",
        hook_err_msg
    );

    for (const auto i : c10::irange(1, forward_args.size())) {
      if (*forward_args[i].type() != *input_tuple_types[i - 1]) {
        TORCH_CHECK(
            false,
            hook_id,
            "has the wrong inner types for the input tuple argument. Received type: '",
            input_arg.type()->annotation_str(),
            "'.\n",
            hook_err_msg
        );
      }
    }
  }
}

void ClassType::checkForwardPreHookSchema(
    int pre_hook_idx,
    const FunctionSchema& pre_hook_schema) const {
  const torch::jit::Function* pre_hook = forward_pre_hooks_[pre_hook_idx];
  std::string hook_id =
      "Pre-hook '" + pre_hook->name() + "' on module '" + name()->name() + "' ";
  std::string pre_hook_err_msg = getForwardPreHookErrorMessage(pre_hook_idx) + "\n";

  // Pre-hooks are expecting two inputs: self, and a Tuple containing the
  // non-self arguments passed to Forward
  TORCH_CHECK(
      pre_hook_schema.arguments().size() == 2,
      hook_id,
      "was expected to only have exactly 2 inputs but it had ",
      pre_hook_schema.arguments().size(),
      " inputs. ",
      pre_hook_err_msg
   );

  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  const std::vector<Argument>& forward_args = forward_schema.arguments();
  checkForwardHookInputArguments(forward_schema, pre_hook_schema, hook_id, pre_hook_err_msg);

  // check return type, expected to be either None, the same type as the input,
  // or the contained single type if the input was a tuple containing a single
  // type.
  TORCH_CHECK(
            !pre_hook_schema.returns().empty(),
            hook_id,
            "is missing a return annotation. Return annotations are required, please add one.\n",
            pre_hook_err_msg
  );
  const Argument return_arg = pre_hook_schema.returns()[0];
  std::string wrong_type_returned_err_msg = hook_id +
      "returned the wrong type of: '" +
      return_arg.type()->annotation_str() + "'.";

  if (return_arg.type()->kind() == NoneType::get()->kind()) {
    return;
  }
  if (forward_args.size() == 2 && *forward_args[1].type() == *return_arg.type()) {
    // TORCH_CHECK below is for the edge case where forward's input is a tuple and the
    // pre-hook returns a matching tuple. Eager doesn't support this- the working eager return
    // for a tuple type is the forward's input tuple wrapped inside of another tuple.
    TORCH_CHECK(
        return_arg.type()->cast<TupleType>() == nullptr,
        wrong_type_returned_err_msg,
        " When forward has a single tuple input argument, the return needs",
        " to be 'None' or a nested tuple containing forward's input tuple",
        " argument as in: 'Tuple[",
        forward_args[1].type()->annotation_str(),
        "]'.\n",
        pre_hook_err_msg
    );
    return;
  }
  // return can only be tuple of nested types now
  // check to make sure return is of tuple type
  TORCH_CHECK(
      return_arg.type()->cast<TupleType>() != nullptr,
      wrong_type_returned_err_msg,
      pre_hook_err_msg
  );
  const at::ArrayRef<TypePtr> return_tuple_types =
      return_arg.type()->castRaw<TupleType>()->elements();
  // check for edge case of Tuple[()] for when forward has no arguments
  if (forward_args.size() == 1) {
    TORCH_CHECK(
        return_tuple_types.empty(),
        wrong_type_returned_err_msg,
        " Was expecting either 'None' or 'Tuple[()]' since forward had ",
        "no arguments.\n",
        pre_hook_err_msg
    );
    return;
  }

  // check that tuple has proper number of contained types
  TORCH_CHECK(
      return_tuple_types.size() == forward_args.size() - 1,
      wrong_type_returned_err_msg,
      " The returned tuple contains the wrong number of contained types.\n",
      pre_hook_err_msg
  );
  // check that contained types match forward types
  for (const auto i : c10::irange(1, forward_args.size())) {
    if (*forward_args[i].type() != *return_tuple_types[i - 1]) {
      TORCH_CHECK(
          false,
          wrong_type_returned_err_msg,
          " The returned tuple contains the wrong inner types.\n",
          pre_hook_err_msg);
    }
  }
}

void ClassType::checkForwardHookSchema(
      int hook_idx,
      const FunctionSchema& hook_schema) const {
  const torch::jit::Function* hook = forward_hooks_[hook_idx];
  std::string hook_id =
      "Hook '" + hook->name() + "' on module '" + name()->name() + "' ";
  std::string hook_err_msg = getForwardHookErrorMessage(hook_idx) + "\n";
  // Hooks are expecting three inputs: self, a Tuple containing the non-self
  // arguments passed to Forward, and the output of either Forward or the
  // previous hook
  TORCH_CHECK(
      hook_schema.arguments().size() == 3,
      hook_id,
      "was expected to only have exactly 3 inputs but it had ",
      hook_schema.arguments().size(),
      " inputs. ",
      hook_err_msg
  );

  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  checkForwardHookInputArguments(forward_schema, hook_schema, hook_id, hook_err_msg);

  // check output tuple
  const Argument& prev_output = (hook_idx == 0)
            ? forward_schema.returns()[0]
            : forward_hooks_[hook_idx - 1]->getSchema().returns()[0];
  const Argument return_arg = hook_schema.arguments()[2];

  // output tuple needs to match prev_output's return exactly
  TORCH_CHECK(
      *prev_output.type() == *return_arg.type(),
      hook_id,
      "has the wrong type for the output argument. Received type: '",
      return_arg.type()->annotation_str(),
      "'. Expected type: '",
      prev_output.type()->annotation_str(),
      "'.\n",
      hook_err_msg
  );
}

torch::jit::Function* ClassType::findMethod(const std::string& name) const {
  for (auto method : methods_) {
    if (name == method->name()) {
      return method;
    }
  }
  return nullptr;
}
torch::jit::Function& ClassType::getMethod(const std::string& name) const {
  auto method = findMethod(name);
  TORCH_CHECK(
      method != nullptr,
      "Couldn't find method: '",
      name,
      "' on class: '",
      repr_str(),
      "'");
  return *method;
}

torch::jit::Function* ClassType::findHook(const std::string& name) const {
  auto hook = findForwardHook(name);
  if (hook == nullptr) {
    hook = findForwardPreHook(name);
  }
  return hook;
}

torch::jit::Function& ClassType::getHook(const std::string& name) const {
  torch::jit::Function* function = findHook(name);
  TORCH_CHECK(
      function != nullptr,
      "Couldn't find: '",
      name,
      "' on class: '",
      repr_str(),
      "'as forward hook or forward pre_hook.");
  return *function;
}

bool ClassType::hasMethod(const std::string& name) const {
  return findMethod(name) != nullptr;
}

void ClassType::addStaticMethod(torch::jit::Function* method) {
  TORCH_CHECK(
      findStaticMethod(method->name()) == nullptr &&
          findMethod(method->name()) == nullptr, "Can't redefine method: ",
      method->name(),
      " on class: ",
      repr_str());
  staticmethods_.emplace_back(method);
}

torch::jit::Function* ClassType::findStaticMethod(const std::string& name) const {
  for (auto method : staticmethods_) {
    if (name == method->name()) {
      return method;
    }
  }
  return nullptr;
}

void ClassType::unsafeRemoveMethod(const std::string& name) {
  size_t slot = 0;
  for (auto method : methods_) {
    if (method->name() == name) {
      methods_.erase(methods_.begin() + slot);
      return;
    }
    slot++;
  }
  TORCH_CHECK(
      false,
      "Can't delete undefined method ",
      name,
      " on class: ",
      repr_str());
}

ClassTypePtr ClassType::refine(at::ArrayRef<TypePtr> refined_slots) const {
  auto ptr = ClassType::create(name(), compilation_unit_, is_module());
  AT_ASSERT(numAttributes() == refined_slots.size());
  for (size_t i = 0; i < attributes_.size(); ++i) {
    AT_ASSERT(refined_slots[i]->isSubtypeOf(*attributes_[i].getType()));
    ptr->addAttribute(attributes_[i].getName(), refined_slots[i], (attributes_[i].getKind() == AttributeKind::PARAMETER),
    (attributes_[i].getKind() == AttributeKind::BUFFER));
  }
  // Copy methods over
  for (const auto& method : methods()) {
    ptr->addMethod(method);
  }
  return ptr;
}

bool ClassType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  if (rhs.castRaw<AnyClassType>()) {
    return true;
  }
  // to improve performance, this check can be cached
  if (auto iface = rhs.cast<InterfaceType>()) {
    // ClassType is not a subtype of InterfaceType if the InterfaceType is a
    // Module Interface Type but the Class Type is not a Module Class Type
    if (!is_module() && iface->is_module()) {
      if (why_not) {
        *why_not << "Class '" << repr_str() << "' is not a subtype of "
                 << "the module interface '" << rhs.repr_str()
                 << "' , only ScriptModule class can be subtype of module"
                 << " interface.\n";
      }
      return false;
    }
    for (const FunctionSchema& schema : iface->methods()) {
      auto self_method = findMethod(schema.name());
      if (!self_method) {
        if (why_not) {
          *why_not << "Class '" << repr_str() << "' does not have method '"
                   << schema.name() << "' but '" << rhs.repr_str()
                   << "' does.\n";
        }
        return false;
      }
      if (!self_method->getSchema().isSubtypeOf(
              // NOLINTNEXTLINE(bugprone-argument-comment)
              schema, /*is_method=*/true, why_not)) {
        if (why_not) {
          *why_not << "Method on class '" << repr_str()
                   << "' (1) is not compatible with interface '"
                   << rhs.repr_str() << "' (2)\n"
                   << "  (1) " << self_method->getSchema() << "\n"
                   << "  (2) " << schema << "\n";
        }
        return false;
      }
    }
    return true;
  }
  return Type::isSubtypeOfExt(rhs, why_not);
}

ClassTypePtr ClassType::create(
    c10::optional<QualifiedName> qualifiedName,
    std::weak_ptr<CompilationUnit> cu,
    bool is_module,
    std::string doc_string,
    std::vector<std::string> unresolved_class_attributes) {
  return ClassTypePtr(new ClassType(
      std::move(qualifiedName),
      std::move(cu),
      is_module,
      std::move(doc_string),
      std::move(unresolved_class_attributes)));
}

ClassType::ClassType(
    c10::optional<QualifiedName> name,
    std::weak_ptr<CompilationUnit> cu,
    bool is_module,
    std::string doc_string,
    std::vector<std::string> unresolved_class_attributes)
    : NamedType(TypeKind::ClassType, std::move(name)),
      compilation_unit_(std::move(cu)),
      isModule_(is_module),
      doc_string_(std::move(doc_string)),
      unresolved_class_attributes_(std::move(unresolved_class_attributes)) {}

const std::vector<torch::jit::Function*>& ClassType::methods() const {
  return methods_;
}

void ClassType::checkNotExist(const std::string& name, const std::string& what) const {
  // Check no overlap with existing constants
  for (size_t i = 0; i < constantNames_.size(); ++i) {
    TORCH_CHECK(
        name != constantNames_[i],
        "attempting to add ",
        what,
        " '",
        name,
        "' to ",
        repr_str(),
        " but a constant field of the same name already exists with value ",
        constantValues_[i]);
  }

  // Check no overlap with existing attributes
  for (const auto & attribute : attributes_) {
    TORCH_CHECK(
        name != attribute.getName(),
        "attempting to add ",
        what,
        " '",
        name,
        "' to ",
        repr_str(),
        " but an attribute field of the same name already exists with type ",
        attribute.getType()->repr_str());
  }
}

void ClassType::addAttribute(ClassAttribute classAttribute) {
    AT_ASSERT(attributes_.size() == attributeTypes_.size());
    attributeTypes_.emplace_back(classAttribute.getType());
    attributes_.emplace_back(std::move(classAttribute));
}

size_t ClassType::addAttribute(
    const std::string& name,
    TypePtr type,
    bool is_parameter,
    bool is_buffer) {
  if (is_parameter && is_buffer){
    TORCH_INTERNAL_ASSERT(false, "Attribute cannot be both a parameter and a buffer!");
  }

  std::string what = is_parameter ? "parameter" : "attribute";
  what += (is_buffer? "buffer" : "not buffer");
  checkNotExist(name, what);

  size_t slot = attributes_.size();

  AttributeKind kind = AttributeKind::REGULAR_ATTRIBUTE;
  if (is_parameter) {
    kind = AttributeKind::PARAMETER;
  } else if (is_buffer) {
    kind = AttributeKind::BUFFER;
  }


  if (is_parameter || is_buffer) {
    TORCH_INTERNAL_ASSERT(is_module(), "adding a parameter or buffer to a non module");
    TORCH_CHECK(
        (type->kind() == TensorType::Kind) ||
            (type->kind() == OptionalType::Kind &&
            type->expectRef<OptionalType>().getElementType()->kind() ==
                TensorType::Kind) ||
            (type->kind() == UnionType::Kind &&
            TensorType::get()->isSubtypeOf(type->expectRef<UnionType>())) ||
            (type->kind() == NoneType::Kind),
        "Expecting parameter or buffer to have either None, Tensor or Optional[Tensor] type, but got: ",
        toString(type));
  }

  addAttribute(ClassAttribute(kind, std::move(type), name));

  return slot;
}

void ClassType::unsafeRemoveAttribute(const std::string& name) {
  auto slot = getAttributeSlot(name);
  attributes_.erase(attributes_.begin() + slot);
  attributeTypes_.erase(attributeTypes_.begin() + slot);
  AT_ASSERT(attributes_.size() == attributeTypes_.size());
}

void ClassType::unsafeChangeAttributeType(const std::string& name, TypePtr new_ty) {
  auto slot = getAttributeSlot(name);
  auto old_attr_info = attributes_[slot];
  AT_ASSERT(old_attr_info.getKind() == AttributeKind::REGULAR_ATTRIBUTE);
  attributes_[slot] = ClassAttribute(old_attr_info.getKind(), new_ty, old_attr_info.getName());
  attributeTypes_[slot] = new_ty;
}

size_t ClassType::addConstant(const std::string& name, const IValue& value) {
  checkNotExist(name, "constant");
  size_t slot = constantNames_.size();
  constantNames_.push_back(name);
  constantValues_.push_back(value);
  return slot;
}

IValue ClassType::getConstant(const std::string& name) const {
  const auto& v = findConstant(name);
  TORCH_CHECK(
      v.has_value(),
      repr_str(),
      " does not have a constant field with name '",
      name,
      "'");
  return *v;
}

IValue ClassType::getConstant(size_t slot) const {
  TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
  TORCH_CHECK(
      slot < constantValues_.size(),
      repr_str(),
      " does not have a constant slot of index ",
      slot);
  return constantValues_[slot];
}

c10::optional<IValue> ClassType::findConstant(const std::string& name) const {
  TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
  size_t pos = 0;
  for (const auto& c : constantNames_) {
    if (name == c) {
      break;
    }
    ++pos;
  }

  if (pos >= constantNames_.size()) {
    return c10::nullopt;
  }
  return constantValues_[pos];
}

void ClassType::unsafeRemoveConstant(const std::string& name) {
  auto slot = getConstantSlot(name);
  constantNames_.erase(constantNames_.begin() + slot);
  constantValues_.erase(constantValues_.begin() + slot);
}

std::shared_ptr<CompilationUnit> ClassType::compilation_unit() {
  auto cu = compilation_unit_.lock();
  return cu;
}

std::shared_ptr<const CompilationUnit> ClassType::compilation_unit() const {
  auto cu = compilation_unit_.lock();
  return cu;
}

c10::optional<ClassType::Property> ClassType::getProperty(const std::string& name) {
  for (auto& prop : properties_) {
    if (name == prop.name) {
      return prop;
    }
  }

  return c10::nullopt;
}

void ClassType::addProperty(const std::string& name, torch::jit::Function* getter, torch::jit::Function* setter) {
  TORCH_INTERNAL_ASSERT(!getProperty(name), "Property named ", name, " already exists!");
  properties_.push_back({name, getter, setter});
}

c10::optional<size_t> ClassType::findConstantSlot(const std::string& name) const {
  TORCH_CHECK(constantNames_.size() == constantValues_.size());
  size_t slot = 0;
  for (const auto& constant : constantNames_) {
    if (name == constant) {
      return slot;
    }
    slot++;
  }
  return c10::nullopt;
}

const std::string& ClassType::getConstantName(size_t slot) const {
  TORCH_CHECK(constantNames_.size() == constantValues_.size());
  TORCH_CHECK(slot < constantNames_.size());
  return constantNames_[slot];
}

size_t ClassType::numConstants() const {
  TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
  return constantNames_.size();
}

at::ArrayRef<IValue> ClassType::constantValues() const {
  return constantValues_;
}

} // namespace c10
