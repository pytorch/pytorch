#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

using namespace torch::jit::script;

void testClassTypeAddRemoveAttr() {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu);
  cls->addAttribute("attr1", TensorType::get());
  cls->addAttribute("attr2", TensorType::get());
  cls->addAttribute("attr3", TensorType::get());
  ASSERT_TRUE(cls->hasAttribute("attr1"));
  ASSERT_TRUE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(cls->hasAttribute("attr3"));

  cls->unsafeRemoveAttribute("attr2");
  ASSERT_TRUE(cls->hasAttribute("attr1"));
  ASSERT_FALSE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(cls->hasAttribute("attr3"));

  cls->unsafeRemoveAttribute("attr1");
  ASSERT_FALSE(cls->hasAttribute("attr1"));
  ASSERT_FALSE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(cls->hasAttribute("attr3"));
}

} // namespace jit
} // namespace torch
