#pragma once

#include <memory>
#include <vector>

#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

#define IS_NODE(T, node)       \
  {                            \
    auto node_ = to<T>(node);  \
    ASSERT_NE(nullptr, node_); \
  }

#define IS_NODE_WITH_NAME(T, node, name) \
  auto name = to<T>(node);               \
  ASSERT_NE(nullptr, name);

#define IS_NODE_WITH_NAME_AND_CAST(T, node, name, Type)        \
  NodePtr<T> name = nullptr;                                   \
  {                                                            \
    auto node_ = to<Cast>(node);                               \
    ASSERT_NE(nullptr, node_);                                 \
    ASSERT_EQ(node_->dtype().scalar_type(), ScalarType::Type); \
    name = to<T>(node_->src_value());                          \
  }                                                            \
  ASSERT_NE(nullptr, name);

#define IS_IMM_WITH_VAL(T, node, val) \
  {                                   \
    auto node_ = to<T##Imm>(node);    \
    ASSERT_NE(nullptr, node_);        \
    ASSERT_EQ(node_->value(), val);   \
  }

#define IS_VAR_WITH_NAME(node, name)     \
  {                                      \
    auto node_ = to<Var>(node);          \
    ASSERT_NE(nullptr, node_);           \
    ASSERT_EQ(node_->name_hint(), name); \
  }

#define IS_BINOP_W_VARS(T, node, name, v1, v2) \
  NodePtr<T> name = nullptr;                   \
  {                                            \
    name = to<T>(node);                        \
    ASSERT_NE(nullptr, name);                  \
    IS_VAR_WITH_NAME(name->lhs(), v1);         \
    IS_VAR_WITH_NAME(name->rhs(), v2);         \
  }

#define IS_BINOP_W_CONST(T, node, name, v, c) \
  NodePtr<T> name = nullptr;                  \
  {                                           \
    name = to<T>(node);                       \
    ASSERT_NE(nullptr, name);                 \
    IS_VAR_WITH_NAME(name->lhs(), v);         \
    IS_IMM_WITH_VAL(Int, name->rhs(), c);     \
  }

#define IS_RAND(node)                   \
  {                                     \
    auto node_ = to<Intrinsics>(node);  \
    ASSERT_NE(nullptr, node_);          \
    ASSERT_EQ(node_->op_type(), kRand); \
  }

void checkIR(StmtPtr s, const std::string& pattern);
void checkExprIR(ExprPtr e, const std::string& pattern);
void checkExprIR(const ExprHandle& e, const std::string& pattern);

} // namespace jit
} // namespace torch
