#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>

#include <ATen/core/alias_info.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

static const std::vector<std::string> subMethodSrcs = {R"JIT(
def one(self, x: Tensor, y: Tensor) -> Tensor:
    return x + y + 1

def forward(self, x: Tensor) -> Tensor:
    return x
)JIT"};
static const std::string parentForward = R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return self.subMod.forward(x)
)JIT";

static constexpr std::string_view moduleInterfaceSrc = R"JIT(
class OneForward(ModuleInterface):
    def one(self, x: Tensor, y: Tensor) -> Tensor:
        pass
    def forward(self, x: Tensor) -> Tensor:
        pass
)JIT";

static void import_libs(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& class_name,
    const std::shared_ptr<Source>& src,
    const std::vector<at::IValue>& tensor_table) {
  SourceImporter si(
      cu,
      &tensor_table,
      [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
      /*version=*/2);
  si.loadType(QualifiedName(class_name));
}

static InterfaceTypePtr makeForwardInterface(
    const char* name,
    const TypePtr& argumentType) {
  auto iface = InterfaceType::create(QualifiedName(name), true);
  iface->addMethod(FunctionSchema(
      "forward",
      "",
      {Argument("self", iface), Argument("x", argumentType)},
      {Argument("", TensorType::get())}));
  return iface;
}

TEST(InterfaceTest, ImportedInterfaceDoesNotOwnItself) {
  std::weak_ptr<CompilationUnit> weakCu;
  std::weak_ptr<InterfaceType> weakInterface;
  {
    auto cu = std::make_shared<CompilationUnit>();
    weakCu = cu;
    std::vector<at::IValue> constants;
    import_libs(
        cu,
        "__torch__.MinimalInterface",
        std::make_shared<Source>(R"JIT(
class MinimalInterface(ModuleInterface):
    def forward(self, x: Tensor) -> Tensor:
        pass
)JIT"),
        constants);
    weakInterface = cu->get_interface("__torch__.MinimalInterface");
  }
  EXPECT_TRUE(weakCu.expired());
  EXPECT_TRUE(weakInterface.expired());
}

TEST(InterfaceTest, ImportedInterfaceIsReleasedWithCompilationUnit) {
  for (const auto* sourceText : {
           R"JIT(
class OneForward(ModuleInterface):
    def forward(self, x: Tensor) -> Tensor:
        pass
)JIT",
           R"JIT(
class OneForward(ModuleInterface):
    def forward(self, x: Optional[Tensor]) -> Optional[Tensor]:
        pass
)JIT"}) {
    SCOPED_TRACE(sourceText);
    std::weak_ptr<CompilationUnit> weakCu;
    std::weak_ptr<InterfaceType> weakInterface;
    std::weak_ptr<OptionalType> weakOptional;
    {
      auto cu = std::make_shared<CompilationUnit>();
      weakCu = cu;
      std::vector<at::IValue> constants;
      import_libs(
          cu,
          "__torch__.OneForward",
          std::make_shared<Source>(sourceText),
          constants);
      auto iface = cu->get_interface("__torch__.OneForward");
      ASSERT_TRUE(iface);
      weakInterface = iface;
      auto schema = iface->getMethodSchema("forward");
      ASSERT_TRUE(schema);
      EXPECT_EQ(schema->arguments().at(0).type().get(), iface.get());
      EXPECT_EQ(schema->arguments().at(0).real_type().get(), iface.get());
      weakOptional = schema->arguments().at(1).type()->cast<OptionalType>();
    }
    EXPECT_TRUE(weakCu.expired());
    EXPECT_TRUE(weakInterface.expired());
    EXPECT_TRUE(weakOptional.expired());
  }
}

TEST(InterfaceTest, ReturnedSchemasKeepInterfaceAlive) {
  std::weak_ptr<CompilationUnit> weakCu;
  std::weak_ptr<InterfaceType> weakInterface;
  std::unique_ptr<FunctionSchema> schema;
  std::vector<FunctionSchema> schemas;
  {
    auto cu = std::make_shared<CompilationUnit>();
    weakCu = cu;
    std::vector<at::IValue> constants;
    import_libs(
        cu,
        "__torch__.OneForward",
        std::make_shared<Source>(moduleInterfaceSrc),
        constants);
    auto iface = cu->get_interface("__torch__.OneForward");
    ASSERT_TRUE(iface);
    weakInterface = iface;
    EXPECT_FALSE(iface->getMethodSchema("missing"));
    schema = iface->getMethodSchema("forward");
    schemas = iface->methodSchemas();
    ASSERT_TRUE(schema);
    EXPECT_EQ(schemas.size(), 2);
  }
  EXPECT_TRUE(weakCu.expired());
  EXPECT_FALSE(weakInterface.expired());
  EXPECT_EQ(schema->arguments().at(0).type().get(), weakInterface.lock().get());
  EXPECT_EQ(
      schemas.at(0).arguments().at(0).real_type().get(),
      weakInterface.lock().get());
  schema.reset();
  EXPECT_FALSE(weakInterface.expired());
  schemas.clear();
  EXPECT_TRUE(weakInterface.expired());
}

TEST(InterfaceTest, ConstAccessorSharesFactoryOwnership) {
  std::weak_ptr<InterfaceType> weakInterface;
  std::unique_ptr<FunctionSchema> schema;
  {
    auto iface =
        InterfaceType::create(QualifiedName("__torch__.ConstReceiver"), true);
    iface->addMethod(
        FunctionSchema("forward", "", {Argument("self", iface)}, {}));
    weakInterface = iface;
    const InterfaceType& constInterface = *iface;
    schema = constInterface.getMethodSchema("forward");
    ASSERT_TRUE(schema);
    const auto receiver =
        schema->arguments().front().type()->cast<InterfaceType>();
    EXPECT_EQ(receiver.get(), iface.get());
    EXPECT_EQ(schema->arguments().front().real_type().get(), iface.get());
    EXPECT_FALSE(receiver.owner_before(iface));
    EXPECT_FALSE(iface.owner_before(receiver));
  }
  EXPECT_FALSE(weakInterface.expired());
  schema.reset();
  EXPECT_TRUE(weakInterface.expired());
}

TEST(InterfaceTest, SharedCopiesUseTheirOwnReceiverIdentity) {
  auto original =
      InterfaceType::create(QualifiedName("__torch__.CopiedReceiver"), true);
  original->addMethod(
      FunctionSchema("forward", "", {Argument("self", original)}, {}));
  auto copied = std::make_shared<InterfaceType>(*original);
  auto assigned =
      InterfaceType::create(QualifiedName("__torch__.AssignedReceiver"), true);
  *assigned = *original;
  std::weak_ptr<InterfaceType> weakOriginal = original;
  original.reset();
  EXPECT_TRUE(weakOriginal.expired());
  for (const auto& iface : {copied, assigned}) {
    const InterfaceType& constInterface = *iface;
    const auto schema = constInterface.getMethodSchema("forward");
    ASSERT_TRUE(schema);
    const auto receiver =
        schema->arguments().front().type()->cast<InterfaceType>();
    EXPECT_EQ(receiver.get(), iface.get());
    EXPECT_EQ(schema->arguments().front().real_type().get(), iface.get());
    EXPECT_FALSE(receiver.owner_before(iface));
    EXPECT_FALSE(iface.owner_before(receiver));
  }
}

TEST(InterfaceTest, MethodNamesOutliveTheirInterface) {
  std::vector<std::string> names;
  std::weak_ptr<InterfaceType> weakInterface;
  {
    auto iface =
        InterfaceType::create(QualifiedName("__torch__.NamedReceiver"), true);
    weakInterface = iface;
    EXPECT_FALSE(iface->hasMethod("missing"));
    EXPECT_TRUE(iface->getMethodNames().empty());
    for (const auto* name :
         {"forward", "method_name_long_enough_to_require_heap_storage"}) {
      iface->addMethod(FunctionSchema(name, "", {Argument("self", iface)}, {}));
    }
    EXPECT_TRUE(iface->hasMethod("forward"));
    EXPECT_TRUE(
        iface->hasMethod("method_name_long_enough_to_require_heap_storage"));
    EXPECT_FALSE(iface->hasMethod("missing"));
    names = iface->getMethodNames();
  }
  EXPECT_TRUE(weakInterface.expired());
  const std::vector<std::string> expected = {
      "forward", "method_name_long_enough_to_require_heap_storage"};
  EXPECT_EQ(names, expected);
}

TEST(InterfaceTest, MethodSchemaRoundTripsArgumentMetadata) {
  c10::AliasInfo selfAlias(false, {"alias::self"}, {"alias::self"});
  c10::AliasInfo nestedAlias(
      false, {"alias::nested_before"}, {"alias::nested_after"});
  c10::AliasInfo outAlias(true, {"alias::out_before"}, {"alias::out_after"});
  outAlias.addContainedType(nestedAlias);
  c10::AliasInfo returnAlias(false, {"alias::result"}, {"alias::result"});
  for (const auto aliasKind :
       {std::optional<c10::AliasAnalysisKind>{},
        std::optional<c10::AliasAnalysisKind>{
            c10::AliasAnalysisKind::FROM_SCHEMA},
        std::optional<c10::AliasAnalysisKind>{
            c10::AliasAnalysisKind::CONSERVATIVE}}) {
    auto iface = InterfaceType::create(
        QualifiedName("__torch__.MetadataReceiver"), true);
    auto original = FunctionSchema(
        "forward",
        "overload",
        {Argument("self", iface, std::nullopt, std::nullopt, false, selfAlias),
         Argument(
             "dtype",
             IntType::get(),
             ScalarTypeType::get(),
             std::nullopt,
             IValue(6)),
         Argument(
             "out", ListType::ofTensors(), 3, std::nullopt, true, outAlias)},
        {Argument(
            "result",
            TensorType::get(),
            std::nullopt,
            std::nullopt,
            false,
            returnAlias)},
        true,
        true);
    if (aliasKind) {
      original.setAliasAnalysis(*aliasKind);
    }
    iface->addMethod(original);
    const auto returned = iface->getMethodSchema("forward");
    ASSERT_TRUE(returned);
    const auto schemas = iface->methodSchemas();
    ASSERT_EQ(schemas.size(), 1);
    const FunctionSchema* results[] = {returned.get(), &schemas.front()};
    for (const auto* schema : results) {
      EXPECT_EQ(schema->name(), "forward");
      EXPECT_EQ(schema->overload_name(), "overload");
      EXPECT_TRUE(schema->is_vararg());
      EXPECT_TRUE(schema->is_varret());
      EXPECT_EQ(schema->isDefaultAliasAnalysisKind(), !aliasKind.has_value());
      EXPECT_EQ(schema->aliasAnalysis(), original.aliasAnalysis());
      ASSERT_EQ(schema->arguments().size(), 3);
      const auto& self = schema->arguments()[0];
      EXPECT_EQ(self.type().get(), iface.get());
      EXPECT_EQ(self.real_type().get(), iface.get());
      ASSERT_NE(self.alias_info(), nullptr);
      EXPECT_EQ(*self.alias_info(), selfAlias);
      const auto& dtype = schema->arguments()[1];
      EXPECT_EQ(dtype.type().get(), IntType::get().get());
      EXPECT_EQ(dtype.real_type().get(), ScalarTypeType::get().get());
      EXPECT_EQ(dtype.default_value(), std::optional<IValue>(IValue(6)));
      EXPECT_FALSE(dtype.kwarg_only());
      const auto& out = schema->arguments()[2];
      EXPECT_EQ(out.N(), std::optional<int32_t>(3));
      EXPECT_FALSE(out.default_value().has_value());
      EXPECT_TRUE(out.kwarg_only());
      EXPECT_TRUE(out.is_out());
      ASSERT_NE(out.alias_info(), nullptr);
      EXPECT_EQ(*out.alias_info(), outAlias);
      ASSERT_EQ(out.alias_info()->containedTypes().size(), 1);
      EXPECT_EQ(out.alias_info()->containedTypes().front(), nestedAlias);
      ASSERT_EQ(schema->returns().size(), 1);
      ASSERT_NE(schema->returns().front().alias_info(), nullptr);
      EXPECT_EQ(*schema->returns().front().alias_info(), returnAlias);
    }
  }
}

TEST(InterfaceTest, CopiedReceiverSchemaAccessRequiresSharedOwnership) {
  const auto iface =
      makeForwardInterface("__torch__.Receiver", TensorType::get());
  const InterfaceType copy(*iface);
  const auto expectOwnershipError = [](const auto& accessSchema) {
    try {
      accessSchema();
      FAIL()
          << "Expected c10::Error for schema access without shared ownership";
    } catch (const c10::Error& error) {
      EXPECT_NE(
          std::string(error.what_without_backtrace()).find("shared ownership"),
          std::string::npos);
    }
  };
  expectOwnershipError([&] { copy.getMethodSchema("forward"); });
  expectOwnershipError([&] { copy.methodSchemas(); });
  EXPECT_TRUE(copy.hasMethod("forward"));
  EXPECT_EQ(copy.getMethodNames(), std::vector<std::string>({"forward"}));
  EXPECT_EQ(copy.getMethodSchema("missing"), nullptr);

  const auto empty =
      InterfaceType::create(QualifiedName("__torch__.Empty"), true);
  const InterfaceType emptyCopy(*empty);
  EXPECT_EQ(emptyCopy.getMethodSchema("missing"), nullptr);
  EXPECT_TRUE(emptyCopy.methodSchemas().empty());
}

TEST(InterfaceTest, SubtypeChecksDoNotMaterializeReceiverSchemas) {
  const auto tensor =
      makeForwardInterface("__torch__.TensorReceiver", TensorType::get());
  const auto integer =
      makeForwardInterface("__torch__.IntReceiver", IntType::get());
  const InterfaceType tensorValue(*tensor);
  const InterfaceType equivalentValue(*tensor);
  const InterfaceType integerValue(*integer);
  EXPECT_TRUE(tensorValue.isSubtypeOfExt(equivalentValue, nullptr));
  EXPECT_FALSE(tensorValue.isSubtypeOfExt(integerValue, nullptr));
  EXPECT_TRUE(tensorValue.equals(equivalentValue));
  EXPECT_FALSE(tensorValue.equals(integerValue));

  auto cu = std::make_shared<CompilationUnit>();
  Module concrete("Concrete", cu);
  concrete.define(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return x
)JIT");
  EXPECT_TRUE(concrete.type()->isSubtypeOfExt(tensorValue, nullptr));
  EXPECT_FALSE(concrete.type()->isSubtypeOfExt(integerValue, nullptr));
}

TEST(InterfaceTest, CopiedReceiverDiagnosticsDoNotRequireSharedOwnership) {
  const auto tensor =
      makeForwardInterface("__torch__.TensorReceiver", TensorType::get());
  const auto integer =
      makeForwardInterface("__torch__.IntReceiver", IntType::get());
  const InterfaceType tensorValue(*tensor);
  const InterfaceType integerValue(*integer);
  const InterfaceType* lhsReceivers[] = {&tensorValue, tensor.get()};
  const InterfaceType* rhsReceivers[] = {&integerValue, integer.get()};
  for (const auto* lhs : lhsReceivers) {
    for (const auto* rhs : rhsReceivers) {
      std::ostringstream whyNot;
      EXPECT_FALSE(lhs->isSubtypeOfExt(*rhs, &whyNot));
      const auto diagnostic = whyNot.str();
      const auto lhsStart = diagnostic.find("(1) forward(");
      const auto rhsStart = diagnostic.find("(2) forward(");
      ASSERT_NE(lhsStart, std::string::npos);
      ASSERT_NE(rhsStart, std::string::npos);
      EXPECT_NE(
          diagnostic.substr(lhsStart, rhsStart - lhsStart)
              .find("TensorReceiver"),
          std::string::npos);
      EXPECT_NE(
          diagnostic.substr(rhsStart).find("IntReceiver"), std::string::npos);
      EXPECT_EQ(diagnostic.find("Any self"), std::string::npos);
    }
  }
  auto cu = std::make_shared<CompilationUnit>();
  Module concrete("Concrete", cu);
  concrete.define(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return x
)JIT");
  for (const auto* rhs : rhsReceivers) {
    std::ostringstream whyNot;
    EXPECT_FALSE(concrete.type()->isSubtypeOfExt(*rhs, &whyNot));
    const auto diagnostic = whyNot.str();
    const auto rhsStart = diagnostic.find("(2) forward(");
    ASSERT_NE(rhsStart, std::string::npos);
    EXPECT_NE(
        diagnostic.substr(rhsStart).find("IntReceiver"), std::string::npos);
    EXPECT_EQ(diagnostic.find("Any self"), std::string::npos);
  }
}

TEST(InterfaceTest, AddMethodRequiresItsOwnReceiver) {
  auto iface = InterfaceType::create(QualifiedName("__torch__.Receiver"), true);
  auto sibling =
      InterfaceType::create(QualifiedName("__torch__.Sibling"), true);
  const auto expectReceiverError = [&](const FunctionSchema& schema,
                                       const char* message) {
    try {
      iface->addMethod(schema);
      FAIL() << "Expected c10::Error for an invalid interface receiver";
    } catch (const c10::Error& error) {
      EXPECT_NE(
          std::string(error.what_without_backtrace()).find(message),
          std::string::npos);
    }
  };
  expectReceiverError(
      FunctionSchema("forward", "", {}, {}), "requires a self argument");
  const FunctionSchema invalidSchemas[] = {
      FunctionSchema("forward", "", {Argument("self", AnyType::get())}, {}),
      FunctionSchema("forward", "", {Argument("self", sibling)}, {}),
      FunctionSchema(
          "forward",
          "",
          {Argument("self", iface, AnyType::get(), std::nullopt, std::nullopt)},
          {}),
      FunctionSchema(
          "forward",
          "",
          {Argument("self", AnyType::get(), iface, std::nullopt, std::nullopt)},
          {})};
  for (const auto& schema : invalidSchemas) {
    expectReceiverError(schema, "type and real_type to be this interface");
  }
  EXPECT_TRUE(iface->getMethodNames().empty());
  iface->addMethod(
      FunctionSchema("forward", "", {Argument("self", iface)}, {}));
  const auto schema = iface->getMethodSchema("forward");
  ASSERT_TRUE(schema);
  EXPECT_EQ(schema->arguments().front().type().get(), iface.get());
  EXPECT_EQ(schema->arguments().front().real_type().get(), iface.get());
}

TEST(InterfaceTest, LongMethodNameMobileSerialization) {
  auto cu = std::make_shared<CompilationUnit>();
  Module parent("parent", cu);
  Module child("child", cu);
  std::vector<at::IValue> constants;
  import_libs(
      cu,
      "__torch__.LongMethod",
      std::make_shared<Source>(R"JIT(
class LongMethod(ModuleInterface):
    def method_name_long_enough_to_require_heap_storage(self, x: Tensor) -> Tensor:
        pass
)JIT"),
      constants);
  child.define(R"JIT(
def method_name_long_enough_to_require_heap_storage(self, x: Tensor) -> Tensor:
    return x + 1
)JIT");
  parent.register_attribute(
      "child", cu->get_interface("__torch__.LongMethod"), child._ivalue());
  parent.define(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return self.child.method_name_long_enough_to_require_heap_storage(x)
)JIT");
  std::stringstream archive;
  parent._save_for_mobile(archive);
  auto reloaded = _load_for_mobile(archive);
  const auto input = torch::ones({2, 3});
  EXPECT_TRUE(reloaded.forward({input}).toTensor().equal(input + 1));
}

TEST(InterfaceTest, ModuleInterfaceSerialization) {
  auto cu = std::make_shared<CompilationUnit>();
  Module parentMod("parentMod", cu);
  Module subMod("subMod", cu);

  std::vector<at::IValue> constantTable;
  import_libs(
      cu,
      "__torch__.OneForward",
      std::make_shared<Source>(moduleInterfaceSrc),
      constantTable);

  for (const std::string& method : subMethodSrcs) {
    subMod.define(method, nativeResolver());
  }
  parentMod.register_attribute(
      "subMod",
      cu->get_interface("__torch__.OneForward"),
      subMod._ivalue(),
      // NOLINTNEXTLINE(bugprone-argument-comment)
      /*is_parameter=*/false);
  parentMod.define(parentForward, nativeResolver());
  ASSERT_TRUE(parentMod.hasattr("subMod"));
  std::stringstream ss;
  parentMod.save(ss);
  Module reloaded_mod = jit::load(ss);
  ASSERT_TRUE(reloaded_mod.hasattr("subMod"));
  InterfaceTypePtr submodType =
      reloaded_mod.type()->getAttribute("subMod")->cast<InterfaceType>();
  ASSERT_TRUE(submodType->is_module());
  const auto input = torch::ones({2, 3});
  EXPECT_TRUE(reloaded_mod.forward({input}).toTensor().equal(input));
  std::stringstream mobileArchive;
  parentMod._save_for_mobile(mobileArchive);
  auto mobileModule = _load_for_mobile(mobileArchive);
  EXPECT_TRUE(mobileModule.forward({input}).toTensor().equal(input));
}

} // namespace jit
} // namespace torch
