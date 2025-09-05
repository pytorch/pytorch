#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <functional>

#include <torch/csrc/utils/python_arg_parser.h>

// Used for Python initialization
#include <pybind11/embed.h>

using namespace torch::test;

void torch_warn_once_A() {
  TORCH_WARN_ONCE("warn once");
}

void torch_warn_once_B() {
  TORCH_WARN_ONCE("warn something else once");
}

void torch_warn() {
  TORCH_WARN("warn multiple times");
}

TEST(UtilsTest, WarnOnce) {
  {
    WarningCapture warnings;

    torch_warn_once_A();
    torch_warn_once_A();
    torch_warn_once_B();
    torch_warn_once_B();

    ASSERT_EQ(count_substr_occurrences(warnings.str(), "warn once"), 1);
    ASSERT_EQ(
        count_substr_occurrences(warnings.str(), "warn something else once"),
        1);
  }
  {
    WarningCapture warnings;

    torch_warn();
    torch_warn();
    torch_warn();

    ASSERT_EQ(
        count_substr_occurrences(warnings.str(), "warn multiple times"), 3);
  }
}

TEST(NoGradTest, SetsGradModeCorrectly) {
  torch::manual_seed(0);
  torch::NoGradGuard guard;
  torch::nn::Linear model(5, 2);
  auto x = torch::randn({10, 5}, torch::requires_grad());
  auto y = model->forward(x);
  torch::Tensor s = y.sum();

  // Mimicking python API behavior:
  ASSERT_THROWS_WITH(
      s.backward(),
      "element 0 of tensors does not require grad and does not have a grad_fn")
}

struct AutogradTest : torch::test::SeedingFixture {
  AutogradTest() {
    x = torch::randn({3, 3}, torch::requires_grad());
    y = torch::randn({3, 3});
    z = x * y;
  }
  torch::Tensor x, y, z;
};

TEST_F(AutogradTest, CanTakeDerivatives) {
  z.backward(torch::ones_like(z));
  ASSERT_TRUE(x.grad().allclose(y));
}

TEST_F(AutogradTest, CanTakeDerivativesOfZeroDimTensors) {
  z.sum().backward();
  ASSERT_TRUE(x.grad().allclose(y));
}

TEST_F(AutogradTest, CanPassCustomGradientInputs) {
  z.sum().backward(torch::ones({}) * 2);
  ASSERT_TRUE(x.grad().allclose(y * 2));
}

TEST(UtilsTest, AmbiguousOperatorDefaults) {
  auto tmp = at::empty({}, at::kCPU);
  at::_test_ambiguous_defaults(tmp);
  at::_test_ambiguous_defaults(tmp, 1);
  at::_test_ambiguous_defaults(tmp, 1, 1);
  at::_test_ambiguous_defaults(tmp, 2, "2");
}

int64_t get_first_element(c10::OptionalIntArrayRef arr) {
  return arr.value()[0];
}

TEST(OptionalArrayRefTest, DanglingPointerFix) {
  // Ensure that the converting constructor of `OptionalArrayRef` does not
  // create a dangling pointer when given a single value
  ASSERT_TRUE(get_first_element(300) == 300);
  ASSERT_TRUE(get_first_element({400}) == 400);
}

TEST(TestPythonArgParser, ParseOneIntListArg) {
  // Test PythonArgParser with one signature that has one argumen of type:
  // INT_LIST || SYM_INT_LIST.

  // Initialize Python using pybind11 scoped interpreter
  pybind11::scoped_interpreter guard{};

  // // Import torch module to initialize PyTorch Python bindings
  py::module::import("torch");

  // the reshape method was chosen at random
  // What matters is there single SymIntArrayRef argument
  torch::PythonArgParser parser({"reshape(SymIntArrayRef shape)"});

  // Verify signature was parsed correctly
  ASSERT_EQ(parser.get_signatures().size(), 1);

  // construct test tensor to test reshape argparser with
  at::Tensor tensor = at::ones({4, 3});
  py::object tensor_obj = py::cast(tensor);

  PyObject* kwargs = nullptr;
  torch::ParsedArgs<1> dst;

  // Test parse method with valid tuple argument: reshape((2, 6))
  PyObject* shape_obj = PyTuple_New(2);
  PyTuple_SetItem(shape_obj, 0, PyLong_FromLong(2));
  PyTuple_SetItem(shape_obj, 1, PyLong_FromLong(6));

  PyObject* args = PyTuple_New(1);
  PyTuple_SetItem(args, 0, shape_obj);

  auto result_tuple = parser.parse<1>(tensor_obj.ptr(), args, kwargs, dst);
  ASSERT_EQ(typeid(result_tuple), typeid(torch::PythonArgs));

  // Test parse method with valid variable int arguments: reshape(2, 6)
  args = PyTuple_New(2);
  PyTuple_SetItem(args, 0, PyLong_FromLong(2));
  PyTuple_SetItem(args, 1, PyLong_FromLong(6));

  auto result_vargs = parser.parse<1>(tensor_obj.ptr(), args, kwargs, dst);
  ASSERT_EQ(typeid(result_vargs), typeid(torch::PythonArgs));

  // Test failure of parse method with invalid variable arguments:
  // reshape((2, 6), "string")
  // This used to run without an error, causing 'silent bugs'.
  shape_obj = PyTuple_New(2);
  PyTuple_SetItem(shape_obj, 0, PyLong_FromLong(2));
  PyTuple_SetItem(shape_obj, 1, PyLong_FromLong(6));

  args = PyTuple_New(2);
  PyTuple_SetItem(args, 0, shape_obj);
  PyTuple_SetItem(args, 1, PyUnicode_FromString("This is an invalid argument"));

  ASSERT_THROW(
      parser.parse<1>(tensor_obj.ptr(), args, kwargs, dst), c10::Error);
}