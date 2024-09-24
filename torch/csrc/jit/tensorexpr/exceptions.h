#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

#include <stdexcept>

// Forward declarations of types

namespace torch::jit::tensorexpr {
class Expr;
class Stmt;
} // namespace torch::jit::tensorexpr

// Forward declarations of functions
namespace std {
TORCH_API std::string to_string(const torch::jit::tensorexpr::ExprPtr&);
TORCH_API std::string to_string(const torch::jit::tensorexpr::StmtPtr&);
} // namespace std

namespace torch::jit::tensorexpr {

class unsupported_dtype : public std::runtime_error {
 public:
  explicit unsupported_dtype() : std::runtime_error("UNSUPPORTED DTYPE") {}
  explicit unsupported_dtype(const std::string& err)
      : std::runtime_error("UNSUPPORTED DTYPE: " + err) {}
};

class out_of_range_index : public std::runtime_error {
 public:
  explicit out_of_range_index() : std::runtime_error("OUT OF RANGE INDEX") {}
  explicit out_of_range_index(const std::string& err)
      : std::runtime_error("OUT OF RANGE INDEX: " + err) {}
};

class unimplemented_lowering : public std::runtime_error {
 public:
  explicit unimplemented_lowering()
      : std::runtime_error("UNIMPLEMENTED LOWERING") {}
  explicit unimplemented_lowering(const ExprPtr& expr)
      : std::runtime_error("UNIMPLEMENTED LOWERING: " + std::to_string(expr)) {}
  explicit unimplemented_lowering(const StmtPtr& stmt)
      : std::runtime_error("UNIMPLEMENTED LOWERING: " + std::to_string(stmt)) {}
};

class malformed_input : public std::runtime_error {
 public:
  explicit malformed_input() : std::runtime_error("MALFORMED INPUT") {}
  explicit malformed_input(const std::string& err)
      : std::runtime_error("MALFORMED INPUT: " + err) {}
  explicit malformed_input(const ExprPtr& expr)
      : std::runtime_error("MALFORMED INPUT: " + std::to_string(expr)) {}
  explicit malformed_input(const std::string& err, const ExprPtr& expr)
      : std::runtime_error(
            "MALFORMED INPUT: " + err + " - " + std::to_string(expr)) {}
  explicit malformed_input(const StmtPtr& stmt)
      : std::runtime_error("MALFORMED INPUT: " + std::to_string(stmt)) {}
  explicit malformed_input(const std::string& err, const StmtPtr& stmt)
      : std::runtime_error(
            "MALFORMED INPUT: " + err + " - " + std::to_string(stmt)) {}
};

class malformed_ir : public std::runtime_error {
 public:
  explicit malformed_ir() : std::runtime_error("MALFORMED IR") {}
  explicit malformed_ir(const std::string& err)
      : std::runtime_error("MALFORMED IR: " + err) {}
  explicit malformed_ir(const ExprPtr& expr)
      : std::runtime_error("MALFORMED IR: " + std::to_string(expr)) {}
  explicit malformed_ir(const std::string& err, const ExprPtr& expr)
      : std::runtime_error(
            "MALFORMED IR: " + err + " - " + std::to_string(expr)) {}
  explicit malformed_ir(const StmtPtr& stmt)
      : std::runtime_error("MALFORMED IR: " + std::to_string(stmt)) {}
  explicit malformed_ir(const std::string& err, const StmtPtr& stmt)
      : std::runtime_error(
            "MALFORMED IR: " + err + " - " + std::to_string(stmt)) {}
};

TORCH_API std::string buildErrorMessage(const std::string& s = "");

} // namespace torch::jit::tensorexpr
