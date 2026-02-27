#include <gtest/gtest.h>
#include <torch/nativert/graph/Serialization.h>

namespace torch::nativert {

// Helper to create tensor argument
torch::_export::TensorArgument makeTensorArg(const std::string& name) {
  torch::_export::TensorArgument arg;
  arg.set_name(name);
  return arg;
}

// Helper to create SymInt argument with name
torch::_export::SymIntArgument makeSymIntArg(const std::string& name) {
  torch::_export::SymIntArgument arg;
  arg.set_as_name(name);
  return arg;
}

// Helper to create SymInt argument with constant int
torch::_export::SymIntArgument makeSymIntArgConst(int64_t value) {
  torch::_export::SymIntArgument arg;
  arg.set_as_int(value);
  return arg;
}

// Helper to create SymBool argument
torch::_export::SymBoolArgument makeSymBoolArg(const std::string& name) {
  torch::_export::SymBoolArgument arg;
  arg.set_as_name(name);
  return arg;
}

// Helper to create OptionalTensorArgument with tensor
torch::_export::OptionalTensorArgument makeOptionalTensorArg(
    const std::string& name) {
  torch::_export::OptionalTensorArgument arg;
  arg.set_as_tensor(makeTensorArg(name));
  return arg;
}

// Helper to create OptionalTensorArgument with None
torch::_export::OptionalTensorArgument makeOptionalTensorArgNone() {
  torch::_export::OptionalTensorArgument arg;
  arg.set_as_none({});
  return arg;
}

TEST(SerializationTest, CheckIsSymbolic) {
  torch::_export::TensorArgument tensor_arg;
  torch::_export::Argument as_tensor_arg;
  as_tensor_arg.set_as_tensor(tensor_arg);
  EXPECT_TRUE(isSymbolic(as_tensor_arg));

  std::vector<torch::_export::TensorArgument> tensor_args;
  torch::_export::Argument as_tensors_arg;
  as_tensors_arg.set_as_tensors(tensor_args);
  EXPECT_TRUE(isSymbolic(as_tensors_arg));

  torch::_export::SymIntArgument sym_int_arg;
  torch::_export::Argument as_sym_int_arg;
  as_sym_int_arg.set_as_sym_int(sym_int_arg);
  EXPECT_TRUE(isSymbolic(as_sym_int_arg));

  torch::_export::Argument as_int_arg;
  as_int_arg.set_as_int(static_cast<int64_t>(1));
  EXPECT_FALSE(isSymbolic(as_int_arg));

  torch::_export::Argument as_bool_arg;
  as_bool_arg.set_as_bool(true);
  EXPECT_FALSE(isSymbolic(as_bool_arg));

  torch::_export::Argument as_string_arg;
  as_string_arg.set_as_string("test_string");
  EXPECT_FALSE(isSymbolic(as_string_arg));
}

// Test isSymbolic for AS_OPTIONAL_TENSOR
TEST(SerializationTest, CheckIsSymbolicOptionalTensor) {
  torch::_export::Argument arg;
  arg.set_as_optional_tensor(makeOptionalTensorArg("opt_tensor"));
  EXPECT_TRUE(isSymbolic(arg));

  // Also test with None optional tensor
  torch::_export::Argument arg_none;
  arg_none.set_as_optional_tensor(makeOptionalTensorArgNone());
  EXPECT_TRUE(isSymbolic(arg_none));
}

// Test isSymbolic for AS_OPTIONAL_TENSORS
TEST(SerializationTest, CheckIsSymbolicOptionalTensors) {
  std::vector<torch::_export::OptionalTensorArgument> opt_tensors;
  opt_tensors.push_back(makeOptionalTensorArg("opt_0"));
  opt_tensors.push_back(makeOptionalTensorArgNone());

  torch::_export::Argument arg;
  arg.set_as_optional_tensors(opt_tensors);
  EXPECT_TRUE(isSymbolic(arg));
}

// Test isSymbolic for AS_SYM_INTS
TEST(SerializationTest, CheckIsSymbolicSymInts) {
  std::vector<torch::_export::SymIntArgument> sym_ints;
  sym_ints.push_back(makeSymIntArg("s0"));
  sym_ints.push_back(makeSymIntArgConst(8));

  torch::_export::Argument arg;
  arg.set_as_sym_ints(sym_ints);
  EXPECT_TRUE(isSymbolic(arg));
}

// Test isSymbolic for AS_SYM_BOOL
TEST(SerializationTest, CheckIsSymbolicSymBool) {
  torch::_export::Argument arg;
  arg.set_as_sym_bool(makeSymBoolArg("sym_bool"));
  EXPECT_TRUE(isSymbolic(arg));
}

// Test isSymbolic for AS_SYM_BOOLS
TEST(SerializationTest, CheckIsSymbolicSymBools) {
  std::vector<torch::_export::SymBoolArgument> sym_bools;
  sym_bools.push_back(makeSymBoolArg("b0"));
  sym_bools.push_back(makeSymBoolArg("b1"));

  torch::_export::Argument arg;
  arg.set_as_sym_bools(sym_bools);
  EXPECT_TRUE(isSymbolic(arg));
}

// Test isSymbolic for AS_SYM_FLOAT
TEST(SerializationTest, CheckIsSymbolicSymFloat) {
  torch::_export::SymFloatArgument sym_float;
  sym_float.set_as_name("sym_float");

  torch::_export::Argument arg;
  arg.set_as_sym_float(sym_float);
  EXPECT_TRUE(isSymbolic(arg));
}

// Test isSymbolic for AS_SYM_FLOATS
TEST(SerializationTest, CheckIsSymbolicSymFloats) {
  torch::_export::SymFloatArgument sym_float_0;
  sym_float_0.set_as_name("sym_float_0");
  torch::_export::SymFloatArgument sym_float_1;
  sym_float_1.set_as_name("sym_float_1");
  std::vector<torch::_export::SymFloatArgument> sym_floats = {
      sym_float_0, sym_float_1};

  torch::_export::Argument arg;
  arg.set_as_sym_floats(sym_floats);
  EXPECT_TRUE(isSymbolic(arg));
}

// Test isSymbolic for AS_CUSTOM_OBJ
TEST(SerializationTest, CheckIsSymbolicCustomObj) {
  torch::_export::CustomObjArgument custom_obj;
  custom_obj.set_name("my_custom_obj");
  custom_obj.set_class_fqn("my.custom.Class");

  torch::_export::Argument arg;
  arg.set_as_custom_obj(custom_obj);
  EXPECT_TRUE(isSymbolic(arg));
}

// Test that non-symbolic types return false
TEST(SerializationTest, CheckIsSymbolicNonSymbolicTypes) {
  // AS_FLOAT
  torch::_export::Argument as_float;
  torch::_export::F64 f64_val;
  f64_val.set(3.14);
  as_float.set_as_float(f64_val);
  EXPECT_FALSE(isSymbolic(as_float));

  // AS_INTS
  torch::_export::Argument as_ints;
  as_ints.set_as_ints({1, 2, 3});
  EXPECT_FALSE(isSymbolic(as_ints));

  // AS_FLOATS
  torch::_export::Argument as_floats;
  torch::_export::F64 f64_1, f64_2, f64_3;
  f64_1.set(1.0);
  f64_2.set(2.0);
  f64_3.set(3.0);
  as_floats.set_as_floats({f64_1, f64_2, f64_3});
  EXPECT_FALSE(isSymbolic(as_floats));

  // AS_BOOLS
  torch::_export::Argument as_bools;
  as_bools.set_as_bools({true, false, true});
  EXPECT_FALSE(isSymbolic(as_bools));

  // AS_NONE
  torch::_export::Argument as_none;
  as_none.set_as_none({});
  EXPECT_FALSE(isSymbolic(as_none));

  // AS_STRINGS
  torch::_export::Argument as_strings;
  as_strings.set_as_strings({"a", "b", "c"});
  EXPECT_FALSE(isSymbolic(as_strings));
}

TEST(SerializationTest, ConstantToValue) {
  torch::_export::Argument as_int_arg;
  as_int_arg.set_as_int(static_cast<int64_t>(42));
  auto value = constantToValue(as_int_arg, false);
  EXPECT_EQ(value, Constant(static_cast<int64_t>(42)));

  torch::_export::Argument as_bool_arg;
  as_bool_arg.set_as_bool(true);
  value = constantToValue(as_bool_arg, false);
  EXPECT_EQ(value, Constant(true));

  torch::_export::Argument as_string_arg;
  as_string_arg.set_as_string("test_string");
  value = constantToValue(as_string_arg, false);
  EXPECT_EQ(value, Constant("test_string"));
}

// Test constantToValue for AS_FLOAT
TEST(SerializationTest, ConstantToValueFloat) {
  torch::_export::Argument arg;
  torch::_export::F64 f64_val;
  f64_val.set(3.14159);
  arg.set_as_float(f64_val);
  auto value = constantToValue(arg, false);
  EXPECT_EQ(value, Constant(3.14159));
}

// Test constantToValue for AS_INTS
TEST(SerializationTest, ConstantToValueInts) {
  torch::_export::Argument arg;
  arg.set_as_ints({1, 2, 3, 4, 5});
  auto value = constantToValue(arg, false);
  std::vector<int64_t> expected = {1, 2, 3, 4, 5};
  EXPECT_EQ(value, Constant(expected));
}

// Test constantToValue for AS_FLOATS
TEST(SerializationTest, ConstantToValueFloats) {
  torch::_export::Argument arg;
  torch::_export::F64 f64_1, f64_2, f64_3;
  f64_1.set(1.0);
  f64_2.set(2.5);
  f64_3.set(3.14);
  arg.set_as_floats({f64_1, f64_2, f64_3});
  auto value = constantToValue(arg, false);
  std::vector<double> expected = {1.0, 2.5, 3.14};
  EXPECT_EQ(value, Constant(expected));
}

// Test constantToValue for AS_BOOLS
TEST(SerializationTest, ConstantToValueBools) {
  torch::_export::Argument arg;
  arg.set_as_bools({true, false, true});
  auto value = constantToValue(arg, false);
  std::vector<bool> expected = {true, false, true};
  EXPECT_EQ(value, Constant(expected));
}

// Test constantToValue for AS_NONE
TEST(SerializationTest, ConstantToValueNone) {
  torch::_export::Argument arg;
  arg.set_as_none({});
  auto value = constantToValue(arg, false);
  EXPECT_EQ(value, Constant(None()));
}

// Test constantToValue for AS_STRINGS
TEST(SerializationTest, ConstantToValueStrings) {
  torch::_export::Argument arg;
  arg.set_as_strings({"hello", "world"});
  auto value = constantToValue(arg, false);
  std::vector<std::string> expected = {"hello", "world"};
  EXPECT_EQ(value, Constant(expected));
}

// Test that symbolic types throw when passed to constantToValue
TEST(SerializationTest, ConstantToValueThrowsOnSymbolicTypes) {
  // AS_TENSOR should throw
  torch::_export::Argument as_tensor;
  as_tensor.set_as_tensor(makeTensorArg("tensor"));
  EXPECT_THROW(constantToValue(as_tensor, false), std::exception);

  // AS_TENSORS should throw
  torch::_export::Argument as_tensors;
  as_tensors.set_as_tensors({makeTensorArg("t1"), makeTensorArg("t2")});
  EXPECT_THROW(constantToValue(as_tensors, false), std::exception);

  // AS_OPTIONAL_TENSORS should throw
  torch::_export::Argument as_opt_tensors;
  as_opt_tensors.set_as_optional_tensors(
      {makeOptionalTensorArg("opt_t1"), makeOptionalTensorArgNone()});
  EXPECT_THROW(constantToValue(as_opt_tensors, false), std::exception);

  // AS_SYM_INT should throw
  torch::_export::Argument as_sym_int;
  as_sym_int.set_as_sym_int(makeSymIntArg("s0"));
  EXPECT_THROW(constantToValue(as_sym_int, false), std::exception);

  // AS_SYM_INTS should throw
  torch::_export::Argument as_sym_ints;
  as_sym_ints.set_as_sym_ints({makeSymIntArg("s0"), makeSymIntArg("s1")});
  EXPECT_THROW(constantToValue(as_sym_ints, false), std::exception);

  // AS_SYM_BOOL should throw
  torch::_export::Argument as_sym_bool;
  as_sym_bool.set_as_sym_bool(makeSymBoolArg("b0"));
  EXPECT_THROW(constantToValue(as_sym_bool, false), std::exception);

  // AS_SYM_BOOLS should throw
  torch::_export::Argument as_sym_bools;
  as_sym_bools.set_as_sym_bools({makeSymBoolArg("b0"), makeSymBoolArg("b1")});
  EXPECT_THROW(constantToValue(as_sym_bools, false), std::exception);

  // AS_CUSTOM_OBJ should throw
  torch::_export::CustomObjArgument custom_obj;
  custom_obj.set_name("obj");
  custom_obj.set_class_fqn("MyClass");
  torch::_export::Argument as_custom_obj;
  as_custom_obj.set_as_custom_obj(custom_obj);
  EXPECT_THROW(constantToValue(as_custom_obj, false), std::exception);

  // AS_OPTIONAL_TENSOR should throw
  torch::_export::Argument as_opt_tensor;
  as_opt_tensor.set_as_optional_tensor(makeOptionalTensorArg("opt"));
  EXPECT_THROW(constantToValue(as_opt_tensor, false), std::exception);
}

} // namespace torch::nativert
