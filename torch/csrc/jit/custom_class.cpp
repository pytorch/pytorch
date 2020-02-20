#include <torch/csrc/jit/custom_class.h>
#include <torch/csrc/jit/script/compilation_unit.h>

#include <atomic>

namespace torch {
namespace jit {

std::vector<c10::RegisterOperators>& registeredOps() {
  static std::vector<c10::RegisterOperators> ops;
  return ops;
}

std::shared_ptr<script::CompilationUnit>& classCU() {
  static std::shared_ptr<script::CompilationUnit> cu =
      std::make_shared<script::CompilationUnit>();
  return cu;
}

bool isCustomClass(const c10::IValue& v) {
  return v.isObject() && v.toObject()->type()->name() &&
      getCustomClass(v.toObject()->type()->name()->qualifiedName());
}

at::TypePtr getCustomClass(const std::string& name) {
  return classCU()->get_type(name);
}

namespace {

std::unordered_map<const detail::RegisteredClassRecord*, ClassTypePtr> &registeredClassTypes() {
  static std::unordered_map<const detail::RegisteredClassRecord*, ClassTypePtr> table;
  return table;
}

void classCallback(const detail::RegisteredClassRecord& clazz) {
  auto classTypePtr =
      ClassType::create(c10::QualifiedName(clazz.qualClassName), classCU());
  classTypePtr->addAttribute("capsule", CapsuleType::get());

  c10::getCustomClassTypeMap().insert({clazz.classTypeidName_intrusive_ptr,
                            c10::StrongTypePtr(classCU(), classTypePtr)});
  c10::getCustomClassTypeMap().insert({clazz.classTypeidName_tagged_capsule,
                            c10::StrongTypePtr(classCU(), classTypePtr)});

  classCU()->register_type(classTypePtr);
  TORCH_INTERNAL_ASSERT(!registeredClassTypes().count(&clazz));
  registeredClassTypes()[&clazz] = classTypePtr;
}

void methodCallback(const detail::RegisteredClassRecord& clazz, const std::string& method_name) {
  const auto& qualClassName = clazz.qualClassName;
  const auto& qualFuncName = clazz.registeredMethods.at(method_name);
  auto graph = std::make_shared<Graph>();

  ensure_c10_registerer_defined();
  auto func_symbol = c10::Symbol::fromQualString(qualFuncName);
  auto ops = torch::jit::getAllOperatorsFor(func_symbol);
  TORCH_CHECK(ops.size() == 1);
  auto &schema = ops[0]->schema();

  for (const auto& arg : schema.arguments()) {
    graph->addInput()->setType(arg.type());
  }

  auto opCall = graph->insertNode(graph->create(
      func_symbol, graph->inputs(), schema.returns().size()));
  Value* res;
  if (schema.returns().size() > 1) {
    const auto& returns = schema.returns();
    size_t op_invocation_idx = 0;
    for (const auto& ret : returns) {
      opCall->output(op_invocation_idx++)->setType(ret.type());
    }
    res = graph->insertNode(graph->createTuple(opCall->outputs()))->output();
  } else if (schema.returns().size() == 1) {
    const auto& returns = schema.returns();
    res = opCall->output()->setType(returns[0].type());
  } else {
    res = graph->insertConstant(IValue())->setType(NoneType::get());
  }
  graph->registerOutput(res);

  auto method = classCU()->create_function(qualClassName + "." + method_name, graph);
  TORCH_INTERNAL_ASSERT(registeredClassTypes().count(&clazz));
  registeredClassTypes().at(&clazz)->addMethod(method);
}

static auto init_jit_custom_class = []() {
  for (auto & kv : registeredClasses()) {
    classCallback(kv.second);
    for (auto & m_kv : kv.second.registeredMethods) {
      methodCallback(kv.second, m_kv.first);
    }
  }
  registerClassRegistrationCallback(classCallback);
  registerMethodRegistrationCallback(methodCallback);
  return 0;
}();

}  // namespace

} // namespace jit
} // namespace torch
