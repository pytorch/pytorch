#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>

#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/import_source.h>
#include <torch/csrc/jit/script/resolver.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

using namespace torch::jit::script;

static const auto classSrcs1 = R"JIT(
class FooNestedTest:
    def __init__(self, y):
        self.y = y

class FooNestedTest2:
    def __init__(self, y):
        self.y = y
        self.nested = __torch__.FooNestedTest(y)

class FooTest:
    def __init__(self, x):
        self.class_attr = __torch__.FooNestedTest(x)
        self.class_attr2 = __torch__.FooNestedTest2(x)
        self.x = self.class_attr.y + self.class_attr2.y
)JIT";

static const auto classSrcs2 = R"JIT(
class FooTest:
    def __init__(self, x):
      self.dx = x
)JIT";

static void import_libs(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& class_name,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& tensor_table) {
  SourceImporter si(
      cu,
      &tensor_table,
      [&](const std::string& name) {
        ASSERT_TRUE(name == "__torch__");
        return src;
      },
      /*version=*/2);
  si.loadNamedType(QualifiedName(class_name));
}

void testClassImport() {
  auto cu1 = std::make_shared<CompilationUnit>();
  auto cu2 = std::make_shared<CompilationUnit>();
  std::vector<at::Tensor> constantTable;
  // Import different versions of FooTest into two namespaces.
  import_libs(
      cu1,
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs1),
      constantTable);
  import_libs(
      cu2,
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs2),
      constantTable);

  // We should get the correct version of `FooTest` for whichever namespace we
  // are referencing
  c10::QualifiedName base("__torch__");
  auto classType1 = cu1->get_class(c10::QualifiedName(base, "FooTest"));
  ASSERT_TRUE(classType1->hasAttribute("x"));
  ASSERT_FALSE(classType1->hasAttribute("dx"));

  auto classType2 = cu2->get_class(c10::QualifiedName(base, "FooTest"));
  ASSERT_TRUE(classType2->hasAttribute("dx"));
  ASSERT_FALSE(classType2->hasAttribute("x"));

  // We should only see FooNestedTest in the first namespace
  auto c = cu1->get_class(c10::QualifiedName(base, "FooNestedTest"));
  ASSERT_TRUE(c);

  c = cu2->get_class(c10::QualifiedName(base, "FooNestedTest"));
  ASSERT_FALSE(c);
}

void testScriptObject() {
  Module m1("m1");
  Module m2("m2");
  std::vector<at::Tensor> constantTable;
  import_libs(
      m1.class_compilation_unit(),
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs1),
      constantTable);
  import_libs(
      m2.class_compilation_unit(),
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs2),
      constantTable);

  // Incorrect arguments for constructor should throw
  c10::QualifiedName base("__torch__");
  ASSERT_ANY_THROW(m1.create_class(c10::QualifiedName(base, "FooTest"), {1}));
  auto x = torch::ones({2, 3});
  auto obj = m2.create_class(c10::QualifiedName(base, "FooTest"), x).toObject();
  auto dx = obj->getAttr("dx");
  ASSERT_TRUE(almostEqual(x, dx.toTensor()));

  auto new_x = torch::rand({2, 3});
  obj->setAttr("dx", new_x);
  auto new_dx = obj->getAttr("dx");
  ASSERT_TRUE(almostEqual(new_x, new_dx.toTensor()));
}

static const auto methodSrc = R"JIT(
def __init__(self, x):
    return x
)JIT";

void testClassDerive() {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu);
  const auto self = SimpleSelf(cls);
  auto methods = cu->define("foo.bar", methodSrc, nativeResolver(), &self);
  auto method = methods[0];
  cls->addAttribute("attr", TensorType::get());
  cls->addMethod(method);
  ASSERT_TRUE(cls->getMethod(method->name()));

  // Refining a new class should retain attributes and methods
  auto newCls = cls->refine({TensorType::get()});
  ASSERT_TRUE(newCls->hasAttribute("attr"));
  ASSERT_TRUE(newCls->getMethod(method->name()));

  auto newCls2 = cls->withContained({TensorType::get()})->expect<ClassType>();
  ASSERT_TRUE(newCls2->hasAttribute("attr"));
  ASSERT_TRUE(newCls2->getMethod(method->name()));
}

} // namespace jit
} // namespace torch
