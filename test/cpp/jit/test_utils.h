#pragma once

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/testing/file_check.h>

#define ASSERT_THROWS_WITH_MESSAGE(statement, substring)                 \
  try {                                                                  \
    (void)statement;                                                     \
    FAIL();                                                              \
  } catch (const std::exception& e) {                                    \
    ASSERT_NE(std::string(e.what()).find(substring), std::string::npos); \
  }

namespace torch {
namespace jit {

using tensor_list = std::vector<at::Tensor>;
using namespace torch::autograd;

// work around the fact that variable_tensor_list doesn't duplicate all
// of std::vector's constructors.
// most constructors are never used in the implementation, just in our tests.
Stack createStack(std::vector<at::Tensor>&& list);

void assertAllClose(const tensor_list& a, const tensor_list& b);

std::vector<at::Tensor> run(
    InterpreterState& interp,
    const std::vector<at::Tensor>& inputs);

std::pair<tensor_list, tensor_list> runGradient(
    Gradient& grad_spec,
    tensor_list& tensors_in,
    tensor_list& tensor_grads_in);

std::shared_ptr<Graph> build_lstm();

at::Tensor t_use(at::Tensor x);
at::Tensor t_def(at::Tensor x);

// given the difference of output vs expected tensor, check whether the
// difference is within a relative tolerance range. This is a standard way of
// matching tensor values up to certain precision
bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs);
bool almostEqual(const at::Tensor& a, const at::Tensor& b);

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b);

std::pair<at::Tensor, at::Tensor> lstm(
    at::Tensor input,
    at::Tensor hx,
    at::Tensor cx,
    at::Tensor w_ih,
    at::Tensor w_hh);

} // namespace jit
} // namespace torch
