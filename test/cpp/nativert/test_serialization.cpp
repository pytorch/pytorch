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
  arg.set_as_none(true);
  return arg;
}

torch::_export::Argument makeTensorArgument(const std::string& name) {
  torch::_export::Argument arg;
  arg.set_as_tensor(makeTensorArg(name));
  return arg;
}

torch::_export::Argument makeIntArgument(int64_t value) {
  torch::_export::Argument arg;
  arg.set_as_int(value);
  return arg;
}

torch::_export::Argument makeStringArgument(const std::string& value) {
  torch::_export::Argument arg;
  arg.set_as_string(value);
  return arg;
}

torch::_export::Argument makeBoolArgument(bool value) {
  torch::_export::Argument arg;
  arg.set_as_bool(value);
  return arg;
}

torch::_export::Argument makeNoneArgument() {
  torch::_export::Argument arg;
  arg.set_as_none(true);
  return arg;
}

torch::_export::Argument makeSymIntArgument(const std::string& name) {
  torch::_export::Argument arg;
  arg.set_as_sym_int(makeSymIntArg(name));
  return arg;
}

torch::_export::ForwardRef<torch::_export::Argument> makeArgumentRef(
    torch::_export::Argument arg) {
  torch::_export::ForwardRef<torch::_export::Argument> ref;
  ref.emplace(std::move(arg));
  return ref;
}

torch::_export::Argument makeTupleArgument(
    std::vector<torch::_export::Argument> args) {
  std::vector<torch::_export::ForwardRef<torch::_export::Argument>> refs;
  refs.reserve(args.size());
  for (auto& arg : args) {
    refs.push_back(makeArgumentRef(std::move(arg)));
  }
  torch::_export::Argument tupleArg;
  tupleArg.set_as_tuple(std::move(refs));
  return tupleArg;
}

torch::_export::InputSpec makeUserInputSpec(torch::_export::Argument arg) {
  torch::_export::UserInputSpec userInput;
  userInput.set_arg(std::move(arg));
  torch::_export::InputSpec inputSpec;
  inputSpec.set_user_input(std::move(userInput));
  return inputSpec;
}

torch::_export::OutputSpec makeUserOutputSpec(torch::_export::Argument arg) {
  torch::_export::UserOutputSpec userOutput;
  userOutput.set_arg(std::move(arg));
  torch::_export::OutputSpec outputSpec;
  outputSpec.set_user_output(std::move(userOutput));
  return outputSpec;
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

TEST(SerializationTest, CheckIsSymbolicTuple) {
  auto tensorTuple =
      makeTupleArgument({makeTensorArgument("x"), makeTensorArgument("y")});
  EXPECT_TRUE(isSymbolic(tensorTuple));

  auto symIntTuple =
      makeTupleArgument({makeSymIntArgument("s0"), makeIntArgument(8)});
  EXPECT_TRUE(isSymbolic(symIntTuple));

  auto intTuple = makeTupleArgument({makeIntArgument(2), makeIntArgument(3)});
  EXPECT_FALSE(isSymbolic(intTuple));

  auto emptyTuple = makeTupleArgument({});
  EXPECT_FALSE(isSymbolic(emptyTuple));
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
  as_ints.set_as_ints(std::vector<int64_t>{1, 2, 3});
  EXPECT_FALSE(isSymbolic(as_ints));

  // AS_FLOATS
  torch::_export::Argument as_floats;
  torch::_export::F64 f64_1, f64_2, f64_3;
  f64_1.set(1.0);
  f64_2.set(2.0);
  f64_3.set(3.0);
  as_floats.set_as_floats(
      std::vector<torch::_export::F64>{f64_1, f64_2, f64_3});
  EXPECT_FALSE(isSymbolic(as_floats));

  // AS_BOOLS
  torch::_export::Argument as_bools;
  as_bools.set_as_bools(std::vector<bool>{true, false, true});
  EXPECT_FALSE(isSymbolic(as_bools));

  // AS_NONE
  torch::_export::Argument as_none;
  as_none.set_as_none(true);
  EXPECT_FALSE(isSymbolic(as_none));

  // AS_STRINGS
  torch::_export::Argument as_strings;
  as_strings.set_as_strings(std::vector<std::string>{"a", "b", "c"});
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

TEST(SerializationTest, ConstantToValueTuple) {
  auto intTuple = makeTupleArgument({makeIntArgument(2), makeIntArgument(3)});
  auto value = constantToValue(intTuple, false);
  std::vector<int64_t> expectedInts = {2, 3};
  EXPECT_EQ(value, Constant(expectedInts));

  auto mixedTuple = makeTupleArgument(
      {makeIntArgument(1), makeStringArgument("two"), makeBoolArgument(true)});
  value = constantToValue(mixedTuple, false);
  auto tupleIValue = constantToIValue(value);
  ASSERT_TRUE(tupleIValue.isTuple());
  const auto& elements = tupleIValue.toTupleRef().elements();
  ASSERT_EQ(elements.size(), 3);
  EXPECT_EQ(elements[0].toInt(), 1);
  EXPECT_EQ(elements[1].toStringRef(), "two");
  EXPECT_EQ(elements[2].toBool(), true);

  auto emptyTuple = makeTupleArgument({});
  value = constantToValue(emptyTuple, false);
  tupleIValue = constantToIValue(value);
  ASSERT_TRUE(tupleIValue.isTuple());
  EXPECT_TRUE(tupleIValue.toTupleRef().elements().empty());
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
  arg.set_as_ints(std::vector<int64_t>{1, 2, 3, 4, 5});
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
  arg.set_as_floats(std::vector<torch::_export::F64>{f64_1, f64_2, f64_3});
  auto value = constantToValue(arg, false);
  std::vector<double> expected = {1.0, 2.5, 3.14};
  EXPECT_EQ(value, Constant(expected));
}

// Test constantToValue for AS_BOOLS
TEST(SerializationTest, ConstantToValueBools) {
  torch::_export::Argument arg;
  arg.set_as_bools(std::vector<bool>{true, false, true});
  auto value = constantToValue(arg, false);
  std::vector<bool> expected = {true, false, true};
  EXPECT_EQ(value, Constant(expected));
}

// Test constantToValue for AS_NONE
TEST(SerializationTest, ConstantToValueNone) {
  torch::_export::Argument arg;
  arg.set_as_none(true);
  auto value = constantToValue(arg, false);
  EXPECT_EQ(value, Constant(None()));
}

// Test constantToValue for AS_STRINGS
TEST(SerializationTest, ConstantToValueStrings) {
  torch::_export::Argument arg;
  arg.set_as_strings(std::vector<std::string>{"hello", "world"});
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
  as_tensors.set_as_tensors(std::vector<torch::_export::TensorArgument>{
      makeTensorArg("t1"), makeTensorArg("t2")});
  EXPECT_THROW(constantToValue(as_tensors, false), std::exception);

  // AS_OPTIONAL_TENSORS should throw
  torch::_export::Argument as_opt_tensors;
  as_opt_tensors.set_as_optional_tensors(
      std::vector<torch::_export::OptionalTensorArgument>{
          makeOptionalTensorArg("opt_t1"), makeOptionalTensorArgNone()});
  EXPECT_THROW(constantToValue(as_opt_tensors, false), std::exception);

  // AS_SYM_INT should throw
  torch::_export::Argument as_sym_int;
  as_sym_int.set_as_sym_int(makeSymIntArg("s0"));
  EXPECT_THROW(constantToValue(as_sym_int, false), std::exception);

  // AS_SYM_INTS should throw
  torch::_export::Argument as_sym_ints;
  as_sym_ints.set_as_sym_ints(std::vector<torch::_export::SymIntArgument>{
      makeSymIntArg("s0"), makeSymIntArg("s1")});
  EXPECT_THROW(constantToValue(as_sym_ints, false), std::exception);

  // AS_SYM_BOOL should throw
  torch::_export::Argument as_sym_bool;
  as_sym_bool.set_as_sym_bool(makeSymBoolArg("b0"));
  EXPECT_THROW(constantToValue(as_sym_bool, false), std::exception);

  // AS_SYM_BOOLS should throw
  torch::_export::Argument as_sym_bools;
  as_sym_bools.set_as_sym_bools(std::vector<torch::_export::SymBoolArgument>{
      makeSymBoolArg("b0"), makeSymBoolArg("b1")});
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

TEST(SerializationTest, JsonToGraphTupleTensorInputCreatesListPack) {
  torch::_export::Graph jsonGraph;

  auto xArg = makeTensorArgument("x");
  auto yArg = makeTensorArgument("y");
  jsonGraph.set_inputs(std::vector<torch::_export::Argument>{xArg, yArg});

  torch::_export::Node node;
  node.set_target("some.op.default");

  torch::_export::NamedArgument tupleInput;
  tupleInput.set_name("pair");
  tupleInput.set_arg(
      makeTupleArgument({makeTensorArgument("x"), makeTensorArgument("y")}));
  node.set_inputs(std::vector<torch::_export::NamedArgument>{tupleInput});

  auto outArg = makeTensorArgument("out");
  node.set_outputs(std::vector<torch::_export::Argument>{outArg});
  jsonGraph.set_nodes(std::vector<torch::_export::Node>{node});
  jsonGraph.set_outputs(std::vector<torch::_export::Argument>{outArg});

  torch::_export::GraphSignature sig;
  sig.set_input_specs(std::vector<torch::_export::InputSpec>{
      makeUserInputSpec(xArg), makeUserInputSpec(yArg)});
  sig.set_output_specs(
      std::vector<torch::_export::OutputSpec>{makeUserOutputSpec(outArg)});

  torch::_export::GraphModule graphModule;
  graphModule.set_graph(jsonGraph);
  graphModule.set_signature(sig);

  auto graph = jsonToGraph(graphModule);

  for (const auto& n : graph->nodes()) {
    if (n.target() == "some.op.default") {
      ASSERT_EQ(n.inputs().size(), 1);
      EXPECT_EQ(n.inputs()[0].name, "pair");
      const Value* tupleValue = n.inputs()[0].value;
      EXPECT_EQ(tupleValue->type().kind(), Type::Kind::TensorList);
      ASSERT_NE(tupleValue->producer(), nullptr);
      EXPECT_EQ(tupleValue->producer()->target(), "prim.ListPack");
      ASSERT_EQ(tupleValue->producer()->inputs().size(), 2);
      EXPECT_EQ(tupleValue->producer()->inputs()[0].value->name(), "x");
      EXPECT_EQ(tupleValue->producer()->inputs()[1].value->name(), "y");
      return;
    }
  }
  FAIL() << "Could not find deserialized tuple input node";
}

TEST(SerializationTest, JsonToGraphTupleOptionalTensorInputCreatesListPack) {
  torch::_export::Graph jsonGraph;

  auto xArg = makeTensorArgument("x");
  jsonGraph.set_inputs(std::vector<torch::_export::Argument>{xArg});

  torch::_export::Node node;
  node.set_target("some.op.default");

  torch::_export::NamedArgument tupleInput;
  tupleInput.set_name("optional_pair");
  tupleInput.set_arg(
      makeTupleArgument({makeTensorArgument("x"), makeNoneArgument()}));
  node.set_inputs(std::vector<torch::_export::NamedArgument>{tupleInput});

  auto outArg = makeTensorArgument("out");
  node.set_outputs(std::vector<torch::_export::Argument>{outArg});
  jsonGraph.set_nodes(std::vector<torch::_export::Node>{node});
  jsonGraph.set_outputs(std::vector<torch::_export::Argument>{outArg});

  torch::_export::GraphSignature sig;
  sig.set_input_specs(
      std::vector<torch::_export::InputSpec>{makeUserInputSpec(xArg)});
  sig.set_output_specs(
      std::vector<torch::_export::OutputSpec>{makeUserOutputSpec(outArg)});

  torch::_export::GraphModule graphModule;
  graphModule.set_graph(jsonGraph);
  graphModule.set_signature(sig);

  auto graph = jsonToGraph(graphModule);

  for (const auto& n : graph->nodes()) {
    if (n.target() == "some.op.default") {
      ASSERT_EQ(n.inputs().size(), 1);
      EXPECT_EQ(n.inputs()[0].name, "optional_pair");
      const Value* tupleValue = n.inputs()[0].value;
      EXPECT_EQ(tupleValue->type().kind(), Type::Kind::OptionalTensorList);
      ASSERT_NE(tupleValue->producer(), nullptr);
      EXPECT_EQ(tupleValue->producer()->target(), "prim.ListPack");
      ASSERT_EQ(tupleValue->producer()->inputs().size(), 2);
      EXPECT_EQ(tupleValue->producer()->inputs()[0].value->name(), "x");
      EXPECT_EQ(
          tupleValue->producer()->inputs()[1].value->type().kind(),
          Type::Kind::None);
      return;
    }
  }
  FAIL() << "Could not find deserialized optional tuple input node";
}

TEST(SerializationTest, JsonToGraphUnsupportedMixedTupleInputThrows) {
  torch::_export::Graph jsonGraph;

  auto xArg = makeTensorArgument("x");
  jsonGraph.set_inputs(std::vector<torch::_export::Argument>{xArg});

  torch::_export::Node node;
  node.set_target("some.op.default");

  torch::_export::NamedArgument tupleInput;
  tupleInput.set_name("mixed_pair");
  tupleInput.set_arg(
      makeTupleArgument({makeTensorArgument("x"), makeIntArgument(1)}));
  node.set_inputs(std::vector<torch::_export::NamedArgument>{tupleInput});

  auto outArg = makeTensorArgument("out");
  node.set_outputs(std::vector<torch::_export::Argument>{outArg});
  jsonGraph.set_nodes(std::vector<torch::_export::Node>{node});
  jsonGraph.set_outputs(std::vector<torch::_export::Argument>{outArg});

  torch::_export::GraphSignature sig;
  sig.set_input_specs(
      std::vector<torch::_export::InputSpec>{makeUserInputSpec(xArg)});
  sig.set_output_specs(
      std::vector<torch::_export::OutputSpec>{makeUserOutputSpec(outArg)});

  torch::_export::GraphModule graphModule;
  graphModule.set_graph(jsonGraph);
  graphModule.set_signature(sig);

  EXPECT_THROW(jsonToGraph(graphModule), std::exception);
}

TEST(SerializationTest, JsonToGraphHigherOrderEmptyTupleCreatesListPack) {
  torch::_export::Graph jsonGraph;

  auto xArg = makeTensorArgument("x");
  jsonGraph.set_inputs(std::vector<torch::_export::Argument>{xArg});

  torch::_export::Node node;
  node.set_target("torch.ops.higher_order.while_loop");

  torch::_export::NamedArgument carriedInput;
  carriedInput.set_name("");
  carriedInput.set_arg(makeTupleArgument({makeTensorArgument("x")}));

  torch::_export::NamedArgument additionalInput;
  additionalInput.set_name("");
  additionalInput.set_arg(makeTupleArgument({}));

  node.set_inputs(std::vector<torch::_export::NamedArgument>{
      carriedInput, additionalInput});

  auto outArg = makeTensorArgument("out");
  node.set_outputs(std::vector<torch::_export::Argument>{outArg});
  jsonGraph.set_nodes(std::vector<torch::_export::Node>{node});
  jsonGraph.set_outputs(std::vector<torch::_export::Argument>{outArg});

  torch::_export::GraphSignature sig;
  sig.set_input_specs(
      std::vector<torch::_export::InputSpec>{makeUserInputSpec(xArg)});
  sig.set_output_specs(
      std::vector<torch::_export::OutputSpec>{makeUserOutputSpec(outArg)});

  torch::_export::GraphModule graphModule;
  graphModule.set_graph(jsonGraph);
  graphModule.set_signature(sig);

  auto graph = jsonToGraph(graphModule);

  for (const auto& n : graph->nodes()) {
    if (n.target() == "torch.ops.higher_order.while_loop") {
      ASSERT_EQ(n.inputs().size(), 2);
      const Value* additionalValue = n.inputs()[1].value;
      EXPECT_EQ(additionalValue->type().kind(), Type::Kind::TensorList);
      ASSERT_NE(additionalValue->producer(), nullptr);
      EXPECT_EQ(additionalValue->producer()->target(), "prim.ListPack");
      EXPECT_TRUE(additionalValue->producer()->inputs().empty());
      return;
    }
  }
  FAIL() << "Could not find deserialized higher-order node";
}

TEST(SerializationTest, GraphSignatureTupleNames) {
  torch::_export::GraphSignature storage;
  storage.set_input_specs(std::vector<torch::_export::InputSpec>{
      makeUserInputSpec(makeTupleArgument(
          {makeTensorArgument("x"), makeTensorArgument("y")}))});
  storage.set_output_specs(std::vector<torch::_export::OutputSpec>{
      makeUserOutputSpec(makeTupleArgument(
          {makeTensorArgument("out0"),
           makeTensorArgument("out1"),
           makeNoneArgument()}))});

  GraphSignature signature(storage);

  ASSERT_EQ(signature.userInputs().size(), 2);
  EXPECT_EQ(signature.userInputs()[0], "x");
  EXPECT_EQ(signature.userInputs()[1], "y");

  ASSERT_EQ(signature.userOutputs().size(), 3);
  ASSERT_TRUE(signature.userOutputs()[0].has_value());
  EXPECT_EQ(signature.userOutputs()[0].value(), "out0");
  ASSERT_TRUE(signature.userOutputs()[1].has_value());
  EXPECT_EQ(signature.userOutputs()[1].value(), "out1");
  EXPECT_FALSE(signature.userOutputs()[2].has_value());
}

// Verify that None-typed input values deserialized from JSON have nullptr
// producer. Previously, they were incorrectly assigned the consuming node as
// producer, which caused dangling pointer crashes in cleanupDeadNodes() when
// graph passes like FuseListUnpack destroyed the consuming node.
TEST(SerializationTest, NoneInputValueHasNullProducer) {
  // Build a minimal JSON graph:
  //   graph(%data):
  //     %out = some.op(data=%data, optional_arg=None)
  //     return (%out)
  torch::_export::Graph jsonGraph;

  // Graph input: a tensor named "data"
  torch::_export::Argument graphInput;
  graphInput.set_as_tensor(makeTensorArg("data"));
  jsonGraph.set_inputs(std::vector<torch::_export::Argument>{graphInput});

  // Node: some.op with a tensor input and a None input
  torch::_export::Node node;
  node.set_target("some.op.default");

  torch::_export::NamedArgument dataInput;
  dataInput.set_name("data");
  torch::_export::Argument dataArg;
  dataArg.set_as_tensor(makeTensorArg("data"));
  dataInput.set_arg(dataArg);

  torch::_export::NamedArgument noneInput;
  noneInput.set_name("optional_arg");
  torch::_export::Argument noneArg;
  noneArg.set_as_none(true);
  noneInput.set_arg(noneArg);

  node.set_inputs(
      std::vector<torch::_export::NamedArgument>{dataInput, noneInput});

  // Output: a tensor named "out"
  torch::_export::Argument nodeOutput;
  nodeOutput.set_as_tensor(makeTensorArg("out"));
  node.set_outputs(std::vector<torch::_export::Argument>{nodeOutput});

  jsonGraph.set_nodes(std::vector<torch::_export::Node>{node});

  // Graph output
  torch::_export::Argument graphOutput;
  graphOutput.set_as_tensor(makeTensorArg("out"));
  jsonGraph.set_outputs(std::vector<torch::_export::Argument>{graphOutput});

  // Build signature with proper input/output specs
  torch::_export::GraphSignature sig;

  torch::_export::UserInputSpec userInput;
  userInput.set_arg(graphInput);
  torch::_export::InputSpec inputSpec;
  inputSpec.set_user_input(userInput);
  sig.set_input_specs(std::vector<torch::_export::InputSpec>{inputSpec});

  torch::_export::UserOutputSpec userOutput;
  userOutput.set_arg(graphOutput);
  torch::_export::OutputSpec outputSpec;
  outputSpec.set_user_output(userOutput);
  sig.set_output_specs(std::vector<torch::_export::OutputSpec>{outputSpec});

  torch::_export::GraphModule graphModule;
  graphModule.set_graph(jsonGraph);
  graphModule.set_signature(sig);

  auto graph = jsonToGraph(graphModule);

  // Find the deserialized node and check its None input
  for (const auto& n : graph->nodes()) {
    if (n.target() == "some.op.default") {
      for (const auto& inp : n.inputs()) {
        if (inp.name == "optional_arg") {
          // The None value should have nullptr producer, not the consuming node
          EXPECT_EQ(inp.value->type().kind(), Type::Kind::None);
          EXPECT_EQ(inp.value->producer(), nullptr)
              << "None input value should have nullptr producer, not the "
                 "consuming node. This was fixed to prevent dangling pointer "
                 "crashes in cleanupDeadNodes().";
          return;
        }
      }
    }
  }
  FAIL() << "Could not find the None input 'optional_arg' on 'some.op.default'";
}

} // namespace torch::nativert
