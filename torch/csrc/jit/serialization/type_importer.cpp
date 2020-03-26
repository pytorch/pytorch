#include <torch/csrc/jit/serialization/type_importer.h>

#include <torch/custom_class.h>

namespace torch {
namespace jit {
TypePtr TypeImporter::import(const TypePtr& origType_) {
  auto it = remappedTypes_.find(origType_);
  if (it != remappedTypes_.end()) {
    return it->second;
  }
  // This is a hack to make sure we properly re-map function types. Otherwise,
  // if we have two distinct TypePtrs that hold the same Function*, we will
  // remap them independently (leading to the creation of two remapped
  // Functions). See [remapping followups].
  if (auto origFunctionType = origType_->cast<FunctionType>()) {
    auto it = std::find_if(
        remappedTypes_.cbegin(), remappedTypes_.cend(), [&](const auto& pr) {
          if (auto fn = pr.first->template cast<FunctionType>()) {
            return origFunctionType->function() == fn->function();
          }
          return false;
        });
    if (it != remappedTypes_.cend()) {
      return it->second;
    }
  }
  auto origType = origType_->cast<c10::NamedType>();
  if (origType == nullptr || !origType->name()) {
    // need to rewrite contained maps?
    if (origType_->containedTypes().size()) {
      auto newContained = fmap(
          origType_->containedTypes(),
          [&](const TypePtr& t) { return import(t); });
      auto newType = origType_->withContained(std::move(newContained));
      remappedTypes_.emplace(origType_, newType);
      return newType;
    } else {
      // no need to remap this type
      remappedTypes_.emplace(origType_, origType_);
      return origType_;
    }
  }
  if (getCustomClass(origType->name()->qualifiedName())) {
    // We don't need to remap custom classes, because they are not managed by
    // compilation units
    remappedTypes_.emplace(origType_, origType_);
    return origType_;
  }
  if (auto classType = origType->cast<ClassType>()) {
    return remap(classType);
  } else if (auto functionType = origType->cast<FunctionType>()) {
    return remap(functionType);
  } else if (auto interfaceType = origType->cast<InterfaceType>()) {
    return remap(interfaceType);
  } else if (auto tupleType = origType->cast<TupleType>()) {
    // FIXME this is broken. For NamedTuples we need to perform the same
    // remapping as everywhere else. But it substantially adds to the
    // complexity of the remapping logic, since NamedTuples are structurally
    // typed, unlike everything else.
    //
    // There is an additional unimplemented thing, where we need to remap the
    // type of IValues that are burned into the graph as constants.
    // See [remapping followups]
    remappedTypes_.emplace(origType_, origType_);
    return origType_;
  }
  TORCH_INTERNAL_ASSERT(false, "the if statement above should be exhaustive");
}

TypePtr TypeImporter::remap(const InterfaceTypePtr& origType) {
  auto interfaceName = origType->name().value();
  if (cu_->get_type(interfaceName) != nullptr) {
    interfaceName = cu_->mangle(interfaceName);
  }

  auto newType =
      InterfaceType::create(std::move(interfaceName), origType->is_module());
  cu_->register_type(newType);
  remappedTypes_.emplace(origType, newType);

  auto typeRemapper = [&](TypePtr in) { return import(in); };
  for (const FunctionSchema& schema : origType->methods()) {
    newType->addMethod(schema.cloneWithRemappedTypes(typeRemapper));
  }

  return newType;
}

TypePtr TypeImporter::remap(const FunctionTypePtr& origType) {
  auto functionName = origType->name().value();
  if (cu_->find_function(functionName) != nullptr) {
    functionName = cu_->mangle(functionName);
  }

  auto typeRemapper = [&](TypePtr in) { return import(in); };
  auto origFunction = origType->function();
  auto graph = origFunction->graph()->copy();
  graph->remapTypes(typeRemapper);
  auto schema = origFunction->getSchema().cloneWithRemappedTypes(typeRemapper);
  auto newFunction = cu_->create_function(functionName, graph);
  newFunction->setSchema(std::move(schema));
  auto newType = FunctionType::create(newFunction);
  remappedTypes_.emplace(origType, newType);
  return newType;
}

TypePtr TypeImporter::remap(const ClassTypePtr& origType) {
  // Mangle the class name if necessary. This can happen when we are
  // referencing class types with the same name in two different compilation
  // units.
  auto className = origType->name().value();
  if (className.prefix().empty()) {
    className = c10::QualifiedName("__torch__", className.name());
  }
  if (cu_->get_type(className) != nullptr) {
    className = cu_->mangle(className);
  }

  auto newType = ClassType::create(className, cu_, origType->is_module());

  // Register the new type. Need to do this first because some of the remapping
  // operations below will operate on `origType`.
  cu_->register_type(newType);
  remappedTypes_.emplace(origType, newType);

  for (size_t i = 0; i < origType->numAttributes(); ++i) {
    auto attrType = origType->getAttribute(i);
    attrType = import(attrType);

    bool isParameter = false;
    if (origType->is_module()) {
      isParameter = origType->is_parameter(i);
    }
    newType->addAttribute(origType->getAttributeName(i), attrType, isParameter);
  }

  for (size_t i = 0; i < origType->numConstants(); ++i) {
    IValue constant = origType->getConstant(i);
    auto namedType = constant.type()->cast<c10::NamedType>();
    TORCH_INTERNAL_ASSERT(
        namedType == nullptr || !namedType->name(),
        "Named types not allowed as constants");
    newType->addConstant(origType->getConstantName(i), constant);
  }

  auto typeRemapper = [&](TypePtr in) { return import(in); };
  for (Function* method : origType->methods()) {
    auto graph = method->graph()->copy();
    graph->remapTypes(typeRemapper);
    auto schema = method->getSchema().cloneWithRemappedTypes(typeRemapper);
    const auto newMethodName =
        c10::QualifiedName(*newType->name(), method->name());
    auto copied = cu_->create_function(newMethodName, graph);
    newType->addMethod(copied);
    copied->setSchema(std::move(schema));
  }

  return newType;
}
} // namespace jit
} // namespace torch
