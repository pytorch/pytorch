#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>
#include "test/cpp/jit/test_base.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include <ATen/core/SchemaMatcher.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {

void testSchemaMatching() {
  {
    RegisterOperators reg({
        Operator(
            "aten::test_vartype(t[] a, t b) -> (t)",
            [](Stack* stack) {
              c10::List<double> list;
              double a;
              pop(stack, list, a);
              push(stack, a);
            },
            c10::AliasAnalysisKind::FROM_SCHEMA),
    });
    Module m("m");
    m.define(R"(
      def test(self):
        a = (1.0, 2.0)
        return torch.test_vartype(a, 2.0)
    )");
    auto result = m.run_method("test");
    TORCH_INTERNAL_ASSERT(result.toDouble() == 2.0);

    const std::string error_example = R"JIT(
      def test_2(self):
          a = (1.0, 2.0)
          non_float = (1, 1)
          return torch.test_vartype(a, non_float)
    )JIT";

    std::string err = "";
    try {
      m.define(error_example);
    } catch (const std::exception& e) {
      err = e.what();
    }
    TORCH_INTERNAL_ASSERT(
        err.find("previously matched to type") != std::string::npos);
  }
  {
    RegisterOperators reg({
        Operator(
            "aten::test_vartype2(t a, t[] b) -> (t[])",
            [](Stack* stack) {
              double a;
              c10::List<double> list;
              pop(stack, a, list);
              push(stack, a);
            },
            AliasAnalysisKind::FROM_SCHEMA),
    });
    Module m("m");
    m.define(R"JIT(
      def test(self):
          a = (1.0, 2.0)
          return torch.test_vartype2(3.0, a)
    )JIT");
    auto result = m.run_method("test");
    TORCH_INTERNAL_ASSERT(result.toDouble() == 3.0);

    static const auto error_exam2 = R"JIT(
      def test_2(self):
          a = (1, 2)
          return torch.test_vartype2(3.0, a)
    )JIT";

    std::string err = "";
    try {
      m.define(error_exam2);
    } catch (const std::exception& e) {
      err = e.what();
    }
    TORCH_INTERNAL_ASSERT(
        err.find("previously matched to type") != std::string::npos);
  }
  {
    // Basic test
    const auto schema =
        parseSchema("test::foo(Tensor a, int b, float c) -> Tensor");
    const auto matcher = c10::SchemaMatcher(
        schema, {TensorType::get(), IntType::get(), FloatType::get()}, {});
    ASSERT_TRUE(matcher.isMatch());

    // Check that input types are correctly populated
    ASSERT_EQ(matcher.inputs().size(), 3)
    ASSERT_TRUE(matcher.inputs()[0]->isSubtypeOf(TensorType::get()));
    ASSERT_EQ(*matcher.inputs()[1], *IntType::get());
    ASSERT_EQ(*matcher.inputs()[2], *FloatType::get());

    // Check that output types are correctly populated
    ASSERT_EQ(matcher.outputs().size(), 1)
    ASSERT_TRUE(matcher.outputs()[0]->isSubtypeOf(TensorType::get()));

    // Arguments should map to their corresponding inputs
    ASSERT_EQ(matcher.argToInputs()[0], 0);
    ASSERT_EQ(matcher.argToInputs()[1], 1);
    ASSERT_EQ(matcher.argToInputs()[2], 2);
  }
  {
    // homogenous tuples should be implicitly convertible to lists of the
    // appropriate type
    const auto schema =
        parseSchema("test::foo(Tensor[] list_of_tensors) -> ()");
    const auto matcher = c10::SchemaMatcher(
        schema,
        {TupleType::create({TensorType::get(), TensorType::get()})},
        {});
    ASSERT_TRUE(matcher.isMatch());
    // 0th argument should map to zeroth input.
    ASSERT_EQ(matcher.argToInputs()[0], 0);
    // The matched inputs should represent the type of the
  }
  {
    // Tensor is implicitly convertible to float/int/scalar
    const auto schema = parseSchema("test::foo(float f, int i, Scalar s) -> ()");
    const auto matcher = c10::SchemaMatcher(
        schema, {TensorType::get(), TensorType::get(), TensorType::get()}, {});
    ASSERT_TRUE(matcher.isMatch());

    // The matcher's reported inputs should have the schema's types
    ASSERT_EQ(*matcher.inputs()[0], *FloatType::get());
    ASSERT_EQ(*matcher.inputs()[1], *IntType::get());
    ASSERT_EQ(*matcher.inputs()[2], *NumberType::get());
  }
  {
    // Scalar (also called NumberType) is implicitly convertible to float/int
    const auto schema = parseSchema("test::foo(float f, int i) -> ()");
    ASSERT_TRUE(
        isMatchingSchema(schema, {NumberType::get(), NumberType::get()}, {}));
  }
  {
    // String is implicitly convertible to device
    const auto schema = parseSchema("test::foo(Device device) -> ()");
    ASSERT_TRUE(isMatchingSchema(schema, {StringType::get()}, {}));
  }
  {
    // Type variables should bind correctly.
    const auto schema =
        parseSchema("test::foo(t[] generic_list, t list_element) -> ()");
    const auto match = c10::SchemaMatcher(
        schema, {ListType::create(TensorType::get()), TensorType::get()}, {});

    ASSERT_TRUE(match.isMatch());

    // The matcher's reported inputs should have resolved the free variable
    // types
    ASSERT_TRUE(match.inputs()[0]->isSubtypeOf(ListType::ofTensors()));
    ASSERT_TRUE(match.inputs()[1]->isSubtypeOf(TensorType::get()));

    // Shouldn't match if the concrete type is not consistent
    ASSERT_FALSE(isMatchingSchema(
        schema, {ListType::create(TensorType::get()), IntType::get()}, {}));
  }
  {
    // We should properly report output vartypes if they are bound.
    const auto schema =
        parseSchema("test::foo(t generic_list, u list_element) -> (u, t)");
    const auto match =
        c10::SchemaMatcher(schema, {IntType::get(), FloatType::get()}, {});
    ASSERT_TRUE(match.isMatch());
    ASSERT_EQ(match.inputs().size(), 2);
    ASSERT_EQ(*match.inputs()[0], *IntType::get());
    ASSERT_EQ(*match.inputs()[1], *FloatType::get());

    ASSERT_EQ(match.outputs().size(), 2);
    ASSERT_EQ(*match.outputs()[0], *FloatType::get());
    ASSERT_EQ(*match.outputs()[1], *IntType::get());
  }
  {
    // Test that kwarg-only matching works
    const auto schema = parseSchema(
        "test::foo(Tensor self, Tensor other, * , Scalar alpha) -> ()");
    ASSERT_TRUE(isMatchingSchema(
        schema,
        {TensorType::get(), TensorType::get()},
        {{"alpha", NumberType::get()}}));
    // Shouldn't match if we specify the same type positionally.
    ASSERT_FALSE(isMatchingSchema(
        schema, {TensorType::get(), TensorType::get(), NumberType::get()}, {}));
  }
  {
    // Test that kwargs work, even if they specify positional arguments
    const auto schema = parseSchema("test::foo(float f, int i) -> ()");
    const auto match = c10::SchemaMatcher(
        schema, {}, {{"i", IntType::get()}, {"f", FloatType::get()}});
    ASSERT_TRUE(match.isMatch());
    ASSERT_EQ(match.inputs().size(), 2);
    ASSERT_EQ(*match.inputs()[0], *FloatType::get());
    ASSERT_EQ(*match.inputs()[1], *IntType::get());
  }
  {
    // Test that kwargs with default arguments are not required
    const auto schema = parseSchema(
        "test::foo(Tensor self, Tensor other, * , Scalar alpha=1) -> ()");
    ASSERT_TRUE(
        isMatchingSchema(schema, {TensorType::get(), TensorType::get()}, {}));
    // Should still match if we provide it explicitly
    ASSERT_TRUE(isMatchingSchema(
        schema,
        {TensorType::get(), TensorType::get()},
        {{"alpha", NumberType::get()}}));
  }
  {
    // Test that single int/floats can broadcast to fixed-size lists
    const auto schema = parseSchema(
        "test::foo(Tensor self, int[3] strides, float[5] blahs) -> ()");
    const auto match = c10::SchemaMatcher(
        schema, {TensorType::get(), IntType::get(), FloatType::get()}, {});
    ASSERT_TRUE(match.isMatch());
    ASSERT_EQ(match.inputs().size(), 3);
    ASSERT_TRUE(match.inputs()[1]->isSubtypeOf(ListType::ofInts()));
    ASSERT_TRUE(match.inputs()[2]->isSubtypeOf(ListType::ofFloats()));

    // int shouldn't be broadcastable to float[N] and vice versa
    ASSERT_FALSE(isMatchingSchema(
        schema, {TensorType::get(), FloatType::get(), FloatType::get()}, {}));
    ASSERT_FALSE(isMatchingSchema(
        schema, {TensorType::get(), IntType::get(), IntType::get()}, {}));
  }
  {
    // Test that vararg expansion works
    const auto schema = parseSchema("test::foo(int[] sizes, *, Scalar foo=1) -> ()");
    const auto match = c10::SchemaMatcher(
        schema, {IntType::get(), IntType::get(), IntType::get()}, {});

    ASSERT_EQ(match.argToInputs().size(), 3);
    ASSERT_EQ(match.argToInputs()[0], 0);
    ASSERT_EQ(match.argToInputs()[1], 0);
    ASSERT_EQ(match.argToInputs()[2], 0);
    ASSERT_TRUE(match.inputs()[0]->isSubtypeOf(ListType::ofInts()));

    // Shouldn't work if there are inconsistent types in the positionals
    ASSERT_FALSE(isMatchingSchema(
        schema, {IntType::get(), FloatType::get(), IntType::get()}, {}));
  }
}
} // namespace jit
} // namespace torch
