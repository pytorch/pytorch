#include <gtest/gtest.h>

#include <test/cpp/jit/test_custom_class_registrations.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <iostream>
#include <string>
#include <vector>

namespace torch {
namespace jit {

TEST(CustomClassTest, TorchbindIValueAPI) {
  script::Module m("m");

  // test make_custom_class API
  auto custom_class_obj = make_custom_class<MyStackClass<std::string>>(
      std::vector<std::string>{"foo", "bar"});
  m.define(R"(
    def forward(self, s : __torch__.torch.classes._TorchScriptTesting._StackString):
      return s.pop(), s
  )");

  auto test_with_obj = [&m](IValue obj, std::string expected) {
    auto res = m.run_method("forward", obj);
    auto tup = res.toTuple();
    AT_ASSERT(tup->elements().size() == 2);
    auto str = tup->elements()[0].toStringRef();
    auto other_obj =
        tup->elements()[1].toCustomClass<MyStackClass<std::string>>();
    AT_ASSERT(str == expected);
    auto ref_obj = obj.toCustomClass<MyStackClass<std::string>>();
    AT_ASSERT(other_obj.get() == ref_obj.get());
  };

  test_with_obj(custom_class_obj, "bar");

  // test IValue() API
  auto my_new_stack = c10::make_intrusive<MyStackClass<std::string>>(
      std::vector<std::string>{"baz", "boo"});
  auto new_stack_ivalue = c10::IValue(my_new_stack);

  test_with_obj(new_stack_ivalue, "boo");
}

TEST(CustomClassTest, ScalarTypeClass) {
  script::Module m("m");

  // test make_custom_class API
  auto cc = make_custom_class<ScalarTypeClass>(at::kFloat);
  m.register_attribute("s", cc.type(), cc, false);

  std::ostringstream oss;
  m.save(oss);
  std::istringstream iss(oss.str());
  caffe2::serialize::IStreamAdapter adapter{&iss};
  auto loaded_module = torch::jit::load(iss, torch::kCPU);
}

class TorchBindTestClass : public torch::jit::CustomClassHolder {
 public:
  std::string get() {
    return "Hello, I am your test custom class";
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char class_doc_string[] = R"(
  I am docstring for TorchBindTestClass
  Args:
      What is an argument? Oh never mind, I don't take any.

  Return:
      How would I know? I am just a holder of some meaningless test methods.
  )";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char method_doc_string[] =
    "I am docstring for TorchBindTestClass get_with_docstring method";

namespace {
static auto reg =
    torch::class_<TorchBindTestClass>(
        "_TorchBindTest",
        "_TorchBindTestClass",
        class_doc_string)
        .def("get", &TorchBindTestClass::get)
        .def("get_with_docstring", &TorchBindTestClass::get, method_doc_string);

} // namespace

// Tests DocString is properly propagated when defining CustomClasses.
TEST(CustomClassTest, TestDocString) {
  auto class_type = getCustomClass(
      "__torch__.torch.classes._TorchBindTest._TorchBindTestClass");
  AT_ASSERT(class_type);
  AT_ASSERT(class_type->doc_string() == class_doc_string);

  AT_ASSERT(class_type->getMethod("get").doc_string().empty());
  AT_ASSERT(
      class_type->getMethod("get_with_docstring").doc_string() ==
      method_doc_string);
}

TEST(CustomClassTest, Serialization) {
  script::Module m("m");

  // test make_custom_class API
  auto custom_class_obj = make_custom_class<MyStackClass<std::string>>(
      std::vector<std::string>{"foo", "bar"});
  m.register_attribute(
      "s",
      custom_class_obj.type(),
      custom_class_obj,
      // NOLINTNEXTLINE(bugprone-argument-comment)
      /*is_parameter=*/false);
  m.define(R"(
    def forward(self):
      return self.s.return_a_tuple()
  )");

  auto test_with_obj = [](script::Module& mod) {
    auto res = mod.run_method("forward");
    auto tup = res.toTuple();
    AT_ASSERT(tup->elements().size() == 2);
    auto i = tup->elements()[1].toInt();
    AT_ASSERT(i == 123);
  };

  auto frozen_m = torch::jit::freeze_module(m.clone());

  test_with_obj(m);
  test_with_obj(frozen_m);

  std::ostringstream oss;
  m.save(oss);
  std::istringstream iss(oss.str());
  caffe2::serialize::IStreamAdapter adapter{&iss};
  auto loaded_module = torch::jit::load(iss, torch::kCPU);

  std::ostringstream oss_frozen;
  frozen_m.save(oss_frozen);
  std::istringstream iss_frozen(oss_frozen.str());
  caffe2::serialize::IStreamAdapter adapter_frozen{&iss_frozen};
  auto loaded_frozen_module = torch::jit::load(iss_frozen, torch::kCPU);
}

// --- Torchbind inheritance tests ---

TEST(CustomClassTest, DefBaseRegistersBaseType) {
  // Verify that def_base<> correctly sets the base type on the ClassType.
  auto dogType = c10::getCustomClassType<c10::intrusive_ptr<Dog>>();
  auto animalType = c10::getCustomClassType<c10::intrusive_ptr<Animal>>();
  ASSERT_TRUE(dogType != nullptr);
  ASSERT_TRUE(animalType != nullptr);

  auto dogClass = std::dynamic_pointer_cast<c10::ClassType>(dogType);
  auto animalClass = std::dynamic_pointer_cast<c10::ClassType>(animalType);
  ASSERT_TRUE(dogClass != nullptr);
  ASSERT_TRUE(animalClass != nullptr);
  ASSERT_TRUE(dogClass->baseType() != nullptr);
  EXPECT_EQ(dogClass->baseType().get(), animalClass.get());
}

TEST(CustomClassTest, IsSubtypeOfWithInheritance) {
  auto dogType = c10::getCustomClassType<c10::intrusive_ptr<Dog>>();
  auto catType = c10::getCustomClassType<c10::intrusive_ptr<Cat>>();
  auto animalType = c10::getCustomClassType<c10::intrusive_ptr<Animal>>();

  // Dog is a subtype of Animal.
  EXPECT_TRUE(dogType->isSubtypeOf(*animalType));
  // Cat is a subtype of Animal.
  EXPECT_TRUE(catType->isSubtypeOf(*animalType));
  // Animal is NOT a subtype of Dog.
  EXPECT_FALSE(animalType->isSubtypeOf(*dogType));
  // Dog is NOT a subtype of Cat.
  EXPECT_FALSE(dogType->isSubtypeOf(*catType));
  // Reflexive: Dog is a subtype of itself.
  EXPECT_TRUE(dogType->isSubtypeOf(*dogType));
}

TEST(CustomClassTest, IsSubtypeOfMultiLevel) {
  // Puppy -> Dog -> Animal: Puppy should be a subtype of both Dog and Animal.
  auto puppyType = c10::getCustomClassType<c10::intrusive_ptr<Puppy>>();
  auto dogType = c10::getCustomClassType<c10::intrusive_ptr<Dog>>();
  auto animalType = c10::getCustomClassType<c10::intrusive_ptr<Animal>>();

  EXPECT_TRUE(puppyType->isSubtypeOf(*dogType));
  EXPECT_TRUE(puppyType->isSubtypeOf(*animalType));
  EXPECT_FALSE(animalType->isSubtypeOf(*puppyType));
}

TEST(CustomClassTest, DerivedToBaseIValueConversion) {
  // Wrap a derived class in IValue, then extract as base type.
  auto dog = c10::make_intrusive<Dog>();
  c10::IValue iv(dog);

  // Should succeed: extract Dog IValue as Animal.
  auto asAnimal = iv.toCustomClass<Animal>();
  ASSERT_TRUE(asAnimal != nullptr);
  EXPECT_EQ(asAnimal->speak(), "woof");
}

TEST(CustomClassTest, MultiLevelIValueConversion) {
  // Puppy -> Dog -> Animal: extract Puppy as Animal via IValue.
  auto puppy = c10::make_intrusive<Puppy>();
  c10::IValue iv(puppy);

  auto asAnimal = iv.toCustomClass<Animal>();
  ASSERT_TRUE(asAnimal != nullptr);
  EXPECT_EQ(asAnimal->speak(), "yip");

  auto asDog = iv.toCustomClass<Dog>();
  ASSERT_TRUE(asDog != nullptr);
  EXPECT_EQ(asDog->speak(), "yip");
}

TEST(CustomClassTest, DerivedPassedAsBaseToConstructor) {
  // Simulate the torchbind path: pass a Dog to AnimalShelter(Animal).
  auto dog = c10::make_intrusive<Dog>();
  auto shelter = c10::make_intrusive<AnimalShelter>(dog);
  EXPECT_EQ(shelter->resident_speak(), "woof");

  auto cat = c10::make_intrusive<Cat>();
  auto shelter2 = c10::make_intrusive<AnimalShelter>(cat);
  EXPECT_EQ(shelter2->resident_speak(), "meow");
}

TEST(CustomClassTest, DerivedPassedAsBaseViaIValue) {
  // Simulate the Python/TorchScript path: wrap derived in IValue, then pass to
  // a function expecting the base type via toCustomClass<Base>().
  auto cat = c10::make_intrusive<Cat>();
  c10::IValue iv(cat);

  auto asAnimal = iv.toCustomClass<Animal>();
  ASSERT_TRUE(asAnimal != nullptr);
  EXPECT_EQ(asAnimal->speak(), "meow");
}

TEST(CustomClassTest, InvalidCrossHierarchyConversionThrows) {
  // Cat should NOT be convertible to Dog.
  auto cat = c10::make_intrusive<Cat>();
  c10::IValue iv(cat);

  EXPECT_THROW(iv.toCustomClass<Dog>(), c10::Error);
}

} // namespace jit
} // namespace torch
