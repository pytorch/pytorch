#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/torch.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// https://docs.python.org/3/extending/extending.html
// https://docs.python.org/3/c-api/
// https://mathieu.fenniak.net/embedding-python-tips/

TEST(Debug, Hello) {
  pybind11::scoped_interpreter g;
  pybind11::object o = pybind11::cast(3);
  pybind11::handle h = pybind11::cast(3);
  auto t = pybind11::make_tuple(o, h);
  pybind11::print(t);
}

TEST(Debug, PrintTensor) {
  pybind11::scoped_interpreter g;
  py::module torch = py::module::import("torch");
  at::Tensor tensor = torch::ones(at::IntArrayRef({10, 20}));
  pybind11::object o = pybind11::cast(tensor);
  pybind11::print(o);
}