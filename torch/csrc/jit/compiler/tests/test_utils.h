#ifndef NNC_TESTS_TEST_UTILS_H_INCLUDED__
#define NNC_TESTS_TEST_UTILS_H_INCLUDED__

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "torch/csrc/jit/compiler/include/buffer.h"
#include "torch/csrc/jit/compiler/include/eval.h"
#include "torch/csrc/jit/compiler/include/function.h"
#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/tensor.h"

namespace torch {
namespace jit {
namespace compiler {

template <class T>
class SimpleTensorEvaluator {
 public:
  void evaluate(const Tensor& t, std::vector<T>* output) {
    int ndim = t.ndim();
    std::vector<int> dims;
    int size = 1;
    for (int i = 0; i < ndim; i++) {
      t.dim(i).accept(&expr_eval_);
      int dim = expr_eval_.value().template as<int>();
      dims.push_back(dim);
      size *= dim;
    }
    const Function& func = t.function();
    const Expr& body = func.body();
    eval_func(dims, func, 0, output, body);
  }

 private:
  void eval_func(
      const std::vector<int>& dims,
      const Function& func,
      int level,
      std::vector<T>* output,
      const Expr& body) {
    if (level >= dims.size()) {
      body.accept(&expr_eval_);
      output->push_back(expr_eval_.value().template as<T>());
      return;
    }
    for (int i = 0; i < dims[level]; i++) {
      Expr wrapped_body = Let::make(func.arg(level), Expr(i), body);
      eval_func(dims, func, level + 1, output, wrapped_body);
    }
  }

  SimpleIREvaluator expr_eval_;
};

template <typename U, typename V>
void ExpectAllNear(
    const std::vector<U>& v1,
    const std::vector<U>& v2,
    V threshold,
    const std::string& name = "") {
  ASSERT_EQ(v1.size(), v2.size());
  for (int i = 0; i < v1.size(); i++) {
    EXPECT_NEAR(v1[i], v2[i], threshold)
        << "element index: " << i << ", name: " << name;
  }
}

} // namespace compiler
} // namespace jit
} // namespace torch

#endif // NNC_TESTS_TEST_UTILS_H_INCLUDED__
