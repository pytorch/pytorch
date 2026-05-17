#include <gtest/gtest.h>
#include <torch/nativert/graph/GraphSignature.h>

namespace torch::nativert {

class GraphSignatureTest : public ::testing::Test {
 protected:
  // Member to hold the GraphSignature object
  GraphSignature graph_sig;

  void SetUp() override {
    torch::_export::TensorArgument param_tensor_arg;
    param_tensor_arg.set_name("param");
    torch::_export::InputToParameterSpec param_input_spec;
    param_input_spec.set_arg(param_tensor_arg);
    param_input_spec.set_parameter_name("param");
    torch::_export::InputSpec input_spec_0;
    input_spec_0.set_parameter(param_input_spec);

    torch::_export::TensorArgument input_tensor_arg;
    input_tensor_arg.set_name("input");
    torch::_export::Argument input_arg;
    input_arg.set_as_tensor(input_tensor_arg);
    torch::_export::UserInputSpec user_input_spec;
    user_input_spec.set_arg(input_arg);
    torch::_export::InputSpec input_spec_1;
    input_spec_1.set_user_input(user_input_spec);

    torch::_export::TensorArgument loss_tensor_arg;
    loss_tensor_arg.set_name("loss");
    torch::_export::LossOutputSpec loss_output_spec;
    loss_output_spec.set_arg(loss_tensor_arg);
    torch::_export::OutputSpec output_spec_0;
    output_spec_0.set_loss_output(loss_output_spec);

    torch::_export::TensorArgument output_tensor_arg;
    output_tensor_arg.set_name("output");
    torch::_export::Argument output_arg;
    output_arg.set_as_tensor(output_tensor_arg);
    torch::_export::UserOutputSpec user_output_spec;
    user_output_spec.set_arg(output_arg);
    torch::_export::OutputSpec output_spec_1;
    output_spec_1.set_user_output(user_output_spec);

    torch::_export::GraphSignature mock_storage;
    mock_storage.set_input_specs({input_spec_0, input_spec_1});
    mock_storage.set_output_specs({output_spec_0, output_spec_1});

    // Initialize the GraphSignature object
    graph_sig = GraphSignature(mock_storage);
  }
};

// Helper function to create a tensor argument
torch::_export::TensorArgument makeTensorArg(const std::string& name) {
  torch::_export::TensorArgument arg;
  arg.set_name(name);
  return arg;
}

// Helper function to create a SymInt argument with a name
torch::_export::SymIntArgument makeSymIntArg(const std::string& name) {
  torch::_export::SymIntArgument arg;
  arg.set_as_name(name);
  return arg;
}

// Helper function to create a SymInt argument with a constant int
torch::_export::SymIntArgument makeSymIntArgConst(int64_t value) {
  torch::_export::SymIntArgument arg;
  arg.set_as_int(value);
  return arg;
}

// Helper function to create a SymBool argument with a name
torch::_export::SymBoolArgument makeSymBoolArg(const std::string& name) {
  torch::_export::SymBoolArgument arg;
  arg.set_as_name(name);
  return arg;
}

// Helper function to create an OptionalTensorArgument with a tensor
torch::_export::OptionalTensorArgument makeOptionalTensorArg(
    const std::string& name) {
  torch::_export::OptionalTensorArgument arg;
  arg.set_as_tensor(makeTensorArg(name));
  return arg;
}

// Helper function to create an OptionalTensorArgument with None
torch::_export::OptionalTensorArgument makeOptionalTensorArgNone() {
  torch::_export::OptionalTensorArgument arg;
  arg.set_as_none({});
  return arg;
}

// Test the constructor with a simple GraphSignature
TEST_F(GraphSignatureTest, ConstructorTest) {
  std::vector<std::string_view> expected_params = {"param"};
  EXPECT_EQ(graph_sig.parameters(), expected_params);

  std::vector<std::string> expected_inputs = {"input"};
  EXPECT_EQ(graph_sig.userInputs(), expected_inputs);

  EXPECT_EQ(graph_sig.userInputs().size(), 1);
  EXPECT_EQ(graph_sig.parameters().size(), 1);
  EXPECT_EQ(graph_sig.lossOutput(), "loss");

  std::vector<std::optional<std::string>> expected_outputs = {"output"};
  EXPECT_EQ(graph_sig.userOutputs(), expected_outputs);
}

// Test the replaceAllUses method
TEST_F(GraphSignatureTest, ReplaceAllUsesTest) {
  graph_sig.replaceAllUses("output", "new_output");
  std::vector<std::optional<std::string>> expected_outputs = {"new_output"};
  EXPECT_EQ(graph_sig.userOutputs(), expected_outputs);
}

// Test AS_TENSORS user input handling - each tensor in the list should become
// a separate entry in userInputs
TEST(GraphSignatureInputTypesTest, UserInputAsTensors) {
  // Create list of tensor arguments
  std::vector<torch::_export::TensorArgument> tensor_list;
  tensor_list.push_back(makeTensorArg("tensor_0"));
  tensor_list.push_back(makeTensorArg("tensor_1"));
  tensor_list.push_back(makeTensorArg("tensor_2"));

  torch::_export::Argument input_arg;
  input_arg.set_as_tensors(tensor_list);
  torch::_export::UserInputSpec user_input_spec;
  user_input_spec.set_arg(input_arg);
  torch::_export::InputSpec input_spec;
  input_spec.set_user_input(user_input_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({input_spec});
  storage.set_output_specs({});

  GraphSignature sig(storage);

  // All tensors in the list should be expanded to individual user inputs
  std::vector<std::string> expected_inputs = {
      "tensor_0", "tensor_1", "tensor_2"};
  EXPECT_EQ(sig.userInputs(), expected_inputs);
}

// Test AS_OPTIONAL_TENSOR user input handling
TEST(GraphSignatureInputTypesTest, UserInputAsOptionalTensor) {
  // Create an optional tensor that has a value
  torch::_export::Argument input_arg;
  input_arg.set_as_optional_tensor(makeOptionalTensorArg("opt_tensor"));
  torch::_export::UserInputSpec user_input_spec;
  user_input_spec.set_arg(input_arg);
  torch::_export::InputSpec input_spec;
  input_spec.set_user_input(user_input_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({input_spec});
  storage.set_output_specs({});

  GraphSignature sig(storage);

  std::vector<std::string> expected_inputs = {"opt_tensor"};
  EXPECT_EQ(sig.userInputs(), expected_inputs);
}

// Test AS_OPTIONAL_TENSOR with None value - should be skipped
TEST(GraphSignatureInputTypesTest, UserInputAsOptionalTensorNone) {
  torch::_export::Argument input_arg;
  input_arg.set_as_optional_tensor(makeOptionalTensorArgNone());
  torch::_export::UserInputSpec user_input_spec;
  user_input_spec.set_arg(input_arg);
  torch::_export::InputSpec input_spec;
  input_spec.set_user_input(user_input_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({input_spec});
  storage.set_output_specs({});

  GraphSignature sig(storage);

  // None optional tensor should be skipped
  EXPECT_TRUE(sig.userInputs().empty());
}

// Test AS_OPTIONAL_TENSORS user input handling - mixed tensors and None values
TEST(GraphSignatureInputTypesTest, UserInputAsOptionalTensors) {
  std::vector<torch::_export::OptionalTensorArgument> opt_tensors;
  opt_tensors.push_back(makeOptionalTensorArg("opt_tensor_0"));
  opt_tensors.push_back(
      makeOptionalTensorArgNone()); // None - should be skipped
  opt_tensors.push_back(makeOptionalTensorArg("opt_tensor_2"));

  torch::_export::Argument input_arg;
  input_arg.set_as_optional_tensors(opt_tensors);
  torch::_export::UserInputSpec user_input_spec;
  user_input_spec.set_arg(input_arg);
  torch::_export::InputSpec input_spec;
  input_spec.set_user_input(user_input_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({input_spec});
  storage.set_output_specs({});

  GraphSignature sig(storage);

  // Only non-None tensors should be in user inputs
  std::vector<std::string> expected_inputs = {"opt_tensor_0", "opt_tensor_2"};
  EXPECT_EQ(sig.userInputs(), expected_inputs);
}

// Test AS_SYM_INT user input handling
TEST(GraphSignatureInputTypesTest, UserInputAsSymInt) {
  torch::_export::Argument input_arg;
  input_arg.set_as_sym_int(makeSymIntArg("sym_int_0"));
  torch::_export::UserInputSpec user_input_spec;
  user_input_spec.set_arg(input_arg);
  torch::_export::InputSpec input_spec;
  input_spec.set_user_input(user_input_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({input_spec});
  storage.set_output_specs({});

  GraphSignature sig(storage);

  std::vector<std::string> expected_inputs = {"sym_int_0"};
  EXPECT_EQ(sig.userInputs(), expected_inputs);
}

// Test AS_SYM_INT with constant value - should be skipped
TEST(GraphSignatureInputTypesTest, UserInputAsSymIntConstant) {
  torch::_export::Argument input_arg;
  input_arg.set_as_sym_int(makeSymIntArgConst(42));
  torch::_export::UserInputSpec user_input_spec;
  user_input_spec.set_arg(input_arg);
  torch::_export::InputSpec input_spec;
  input_spec.set_user_input(user_input_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({input_spec});
  storage.set_output_specs({});

  GraphSignature sig(storage);

  // Constant symints should be skipped
  EXPECT_TRUE(sig.userInputs().empty());
}

// Test AS_SYM_INTS user input handling - mixed names and constants
TEST(GraphSignatureInputTypesTest, UserInputAsSymInts) {
  std::vector<torch::_export::SymIntArgument> sym_ints;
  sym_ints.push_back(makeSymIntArg("s0"));
  sym_ints.push_back(makeSymIntArgConst(8)); // Constant - should be skipped
  sym_ints.push_back(makeSymIntArg("s1"));

  torch::_export::Argument input_arg;
  input_arg.set_as_sym_ints(sym_ints);
  torch::_export::UserInputSpec user_input_spec;
  user_input_spec.set_arg(input_arg);
  torch::_export::InputSpec input_spec;
  input_spec.set_user_input(user_input_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({input_spec});
  storage.set_output_specs({});

  GraphSignature sig(storage);

  // Only named symints should be in user inputs
  std::vector<std::string> expected_inputs = {"s0", "s1"};
  EXPECT_EQ(sig.userInputs(), expected_inputs);
}

// Test AS_TENSORS user output handling
TEST(GraphSignatureOutputTypesTest, UserOutputAsTensors) {
  std::vector<torch::_export::TensorArgument> tensor_list;
  tensor_list.push_back(makeTensorArg("out_tensor_0"));
  tensor_list.push_back(makeTensorArg("out_tensor_1"));

  torch::_export::Argument output_arg;
  output_arg.set_as_tensors(tensor_list);
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  std::vector<std::optional<std::string>> expected_outputs = {
      "out_tensor_0", "out_tensor_1"};
  EXPECT_EQ(sig.userOutputs(), expected_outputs);
}

// Test AS_OPTIONAL_TENSOR user output handling
TEST(GraphSignatureOutputTypesTest, UserOutputAsOptionalTensor) {
  torch::_export::Argument output_arg;
  output_arg.set_as_optional_tensor(makeOptionalTensorArg("opt_out_tensor"));
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  std::vector<std::optional<std::string>> expected_outputs = {"opt_out_tensor"};
  EXPECT_EQ(sig.userOutputs(), expected_outputs);
}

// Test AS_OPTIONAL_TENSOR user output with None - should have nullopt name
TEST(GraphSignatureOutputTypesTest, UserOutputAsOptionalTensorNone) {
  torch::_export::Argument output_arg;
  output_arg.set_as_optional_tensor(makeOptionalTensorArgNone());
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  // None optional tensor should result in nullopt
  EXPECT_EQ(sig.userOutputs().size(), 1);
  EXPECT_FALSE(sig.userOutputs()[0].has_value());
}

// Test AS_OPTIONAL_TENSORS user output handling - mixed tensors and None
TEST(GraphSignatureOutputTypesTest, UserOutputAsOptionalTensors) {
  std::vector<torch::_export::OptionalTensorArgument> opt_tensors;
  opt_tensors.push_back(makeOptionalTensorArg("opt_out_0"));
  opt_tensors.push_back(makeOptionalTensorArgNone());
  opt_tensors.push_back(makeOptionalTensorArg("opt_out_2"));

  torch::_export::Argument output_arg;
  output_arg.set_as_optional_tensors(opt_tensors);
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  EXPECT_EQ(sig.userOutputs().size(), 3);
  EXPECT_EQ(sig.userOutputs()[0], "opt_out_0");
  EXPECT_FALSE(sig.userOutputs()[1].has_value()); // None becomes nullopt
  EXPECT_EQ(sig.userOutputs()[2], "opt_out_2");
}

// Test AS_SYM_INT user output handling
TEST(GraphSignatureOutputTypesTest, UserOutputAsSymInt) {
  torch::_export::Argument output_arg;
  output_arg.set_as_sym_int(makeSymIntArg("sym_out_0"));
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  std::vector<std::optional<std::string>> expected_outputs = {"sym_out_0"};
  EXPECT_EQ(sig.userOutputs(), expected_outputs);
}

// Test AS_SYM_INTS user output handling
TEST(GraphSignatureOutputTypesTest, UserOutputAsSymInts) {
  std::vector<torch::_export::SymIntArgument> sym_ints;
  sym_ints.push_back(makeSymIntArg("sym_out_0"));
  sym_ints.push_back(makeSymIntArgConst(10)); // Constant - should be skipped
  sym_ints.push_back(makeSymIntArg("sym_out_1"));

  torch::_export::Argument output_arg;
  output_arg.set_as_sym_ints(sym_ints);
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  // Only named symints should appear
  std::vector<std::optional<std::string>> expected_outputs = {
      "sym_out_0", "sym_out_1"};
  EXPECT_EQ(sig.userOutputs(), expected_outputs);
}

// Test AS_SYM_BOOL user output handling
TEST(GraphSignatureOutputTypesTest, UserOutputAsSymBool) {
  torch::_export::Argument output_arg;
  output_arg.set_as_sym_bool(makeSymBoolArg("sym_bool_out"));
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  std::vector<std::optional<std::string>> expected_outputs = {"sym_bool_out"};
  EXPECT_EQ(sig.userOutputs(), expected_outputs);
}

// Test AS_SYM_BOOLS user output handling
TEST(GraphSignatureOutputTypesTest, UserOutputAsSymBools) {
  std::vector<torch::_export::SymBoolArgument> sym_bools;
  sym_bools.push_back(makeSymBoolArg("sym_bool_0"));
  sym_bools.push_back(makeSymBoolArg("sym_bool_1"));

  torch::_export::Argument output_arg;
  output_arg.set_as_sym_bools(sym_bools);
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  std::vector<std::optional<std::string>> expected_outputs = {
      "sym_bool_0", "sym_bool_1"};
  EXPECT_EQ(sig.userOutputs(), expected_outputs);
}

// Test AS_SYM_FLOAT user output handling - should result in nullopt (unnamed)
TEST(GraphSignatureOutputTypesTest, UserOutputAsSymFloat) {
  torch::_export::SymFloatArgument sym_float;
  sym_float.set_as_name("sym_float_out");
  torch::_export::Argument output_arg;
  output_arg.set_as_sym_float(sym_float);
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  // SymFloat outputs currently result in nullopt (unnamed)
  EXPECT_EQ(sig.userOutputs().size(), 1);
  EXPECT_FALSE(sig.userOutputs()[0].has_value());
}

// Test AS_SYM_FLOATS user output handling
TEST(GraphSignatureOutputTypesTest, UserOutputAsSymFloats) {
  torch::_export::SymFloatArgument sym_float_0;
  sym_float_0.set_as_name("sym_float_0");
  torch::_export::SymFloatArgument sym_float_1;
  sym_float_1.set_as_name("sym_float_1");
  std::vector<torch::_export::SymFloatArgument> sym_floats = {
      sym_float_0, sym_float_1};

  torch::_export::Argument output_arg;
  output_arg.set_as_sym_floats(sym_floats);
  torch::_export::UserOutputSpec user_output_spec;
  user_output_spec.set_arg(output_arg);
  torch::_export::OutputSpec output_spec;
  output_spec.set_user_output(user_output_spec);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec});

  GraphSignature sig(storage);

  // SymFloats outputs are treated as unnamed (nullopt)
  EXPECT_EQ(sig.userOutputs().size(), 2);
  for (const auto& output : sig.userOutputs()) {
    EXPECT_FALSE(output.has_value());
  }
}

// Test mixed input types in the same GraphSignature
TEST(GraphSignatureMixedTypesTest, MixedInputTypes) {
  // First input: single tensor
  torch::_export::Argument input_arg_0;
  input_arg_0.set_as_tensor(makeTensorArg("single_tensor"));
  torch::_export::UserInputSpec user_input_spec_0;
  user_input_spec_0.set_arg(input_arg_0);
  torch::_export::InputSpec input_spec_0;
  input_spec_0.set_user_input(user_input_spec_0);

  // Second input: list of tensors
  std::vector<torch::_export::TensorArgument> tensor_list;
  tensor_list.push_back(makeTensorArg("list_tensor_0"));
  tensor_list.push_back(makeTensorArg("list_tensor_1"));
  torch::_export::Argument input_arg_1;
  input_arg_1.set_as_tensors(tensor_list);
  torch::_export::UserInputSpec user_input_spec_1;
  user_input_spec_1.set_arg(input_arg_1);
  torch::_export::InputSpec input_spec_1;
  input_spec_1.set_user_input(user_input_spec_1);

  // Third input: symint
  torch::_export::Argument input_arg_2;
  input_arg_2.set_as_sym_int(makeSymIntArg("batch_size"));
  torch::_export::UserInputSpec user_input_spec_2;
  user_input_spec_2.set_arg(input_arg_2);
  torch::_export::InputSpec input_spec_2;
  input_spec_2.set_user_input(user_input_spec_2);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({input_spec_0, input_spec_1, input_spec_2});
  storage.set_output_specs({});

  GraphSignature sig(storage);

  std::vector<std::string> expected_inputs = {
      "single_tensor", "list_tensor_0", "list_tensor_1", "batch_size"};
  EXPECT_EQ(sig.userInputs(), expected_inputs);
}

// Test mixed output types in the same GraphSignature
TEST(GraphSignatureMixedTypesTest, MixedOutputTypes) {
  // First output: single tensor
  torch::_export::Argument output_arg_0;
  output_arg_0.set_as_tensor(makeTensorArg("single_out"));
  torch::_export::UserOutputSpec user_output_spec_0;
  user_output_spec_0.set_arg(output_arg_0);
  torch::_export::OutputSpec output_spec_0;
  output_spec_0.set_user_output(user_output_spec_0);

  // Second output: list of tensors
  std::vector<torch::_export::TensorArgument> tensor_list;
  tensor_list.push_back(makeTensorArg("list_out_0"));
  tensor_list.push_back(makeTensorArg("list_out_1"));
  torch::_export::Argument output_arg_1;
  output_arg_1.set_as_tensors(tensor_list);
  torch::_export::UserOutputSpec user_output_spec_1;
  user_output_spec_1.set_arg(output_arg_1);
  torch::_export::OutputSpec output_spec_1;
  output_spec_1.set_user_output(user_output_spec_1);

  // Third output: symint
  torch::_export::Argument output_arg_2;
  output_arg_2.set_as_sym_int(makeSymIntArg("out_size"));
  torch::_export::UserOutputSpec user_output_spec_2;
  user_output_spec_2.set_arg(output_arg_2);
  torch::_export::OutputSpec output_spec_2;
  output_spec_2.set_user_output(user_output_spec_2);

  torch::_export::GraphSignature storage;
  storage.set_input_specs({});
  storage.set_output_specs({output_spec_0, output_spec_1, output_spec_2});

  GraphSignature sig(storage);

  std::vector<std::optional<std::string>> expected_outputs = {
      "single_out", "list_out_0", "list_out_1", "out_size"};
  EXPECT_EQ(sig.userOutputs(), expected_outputs);
}

} // namespace torch::nativert
