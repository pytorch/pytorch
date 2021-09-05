#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

#include <sstream>
#include <stdexcept>

// Forward declarations of types
namespace torch {
namespace jit {
namespace tensorexpr {
class Expr;
class Stmt;
} // namespace tensorexpr
} // namespace jit
} // namespace torch

// Forward declarations of functions
namespace std {
TORCH_API std::string to_string(const torch::jit::tensorexpr::ExprPtr);
TORCH_API std::string to_string(const torch::jit::tensorexpr::StmtPtr);
} // namespace std

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API std::string buildErrorMessage(const std::string& s);

class compilation_error : public c10::Error {
 public:
  explicit compilation_error(const std::string& err)
      : c10::Error(
            {
                __func__,
                __FILE__,
                static_cast<uint32_t>(__LINE__),
            },
            buildErrorMessage(err)) {}
};

class unsupported_dtype : public compilation_error {
 public:
  explicit unsupported_dtype() : compilation_error("UNSUPPORTED DTYPE") {}
  explicit unsupported_dtype(const std::string& err)
      : compilation_error("UNSUPPORTED DTYPE: " + err) {}
};

class out_of_range_index : public compilation_error {
 public:
  explicit out_of_range_index() : compilation_error("OUT OF RANGE INDEX") {}
  explicit out_of_range_index(const std::string& err)
      : compilation_error("OUT OF RANGE INDEX: " + err) {}
};

class unimplemented_lowering : public compilation_error {
 public:
  explicit unimplemented_lowering()
      : compilation_error("UNIMPLEMENTED LOWERING") {}
  explicit unimplemented_lowering(ExprPtr expr)
      : compilation_error("UNIMPLEMENTED LOWERING: " + std::to_string(expr)) {}
  explicit unimplemented_lowering(StmtPtr stmt)
      : compilation_error("UNIMPLEMENTED LOWERING: " + std::to_string(stmt)) {}
};

class malformed_input : public compilation_error {
 public:
  explicit malformed_input() : compilation_error("MALFORMED INPUT") {}
  explicit malformed_input(const std::string& err)
      : compilation_error("MALFORMED INPUT: " + err) {}
  explicit malformed_input(ExprPtr expr)
      : compilation_error("MALFORMED INPUT: " + std::to_string(expr)) {}
  explicit malformed_input(const std::string& err, ExprPtr expr)
      : compilation_error(
            "MALFORMED INPUT: " + err + " - " + std::to_string(expr)) {}
  explicit malformed_input(StmtPtr stmt)
      : compilation_error("MALFORMED INPUT: " + std::to_string(stmt)) {}
  explicit malformed_input(const std::string& err, StmtPtr stmt)
      : compilation_error(
            "MALFORMED INPUT: " + err + " - " + std::to_string(stmt)) {}
};

class malformed_ir : public compilation_error {
 public:
  explicit malformed_ir() : compilation_error("MALFORMED IR") {}
  explicit malformed_ir(const std::string& err)
      : compilation_error("MALFORMED IR: " + err) {}
  explicit malformed_ir(ExprPtr expr)
      : compilation_error("MALFORMED IR: " + std::to_string(expr)) {}
  explicit malformed_ir(const std::string& err, ExprPtr expr)
      : compilation_error(
            "MALFORMED IR: " + err + " - " + std::to_string(expr)) {}
  explicit malformed_ir(StmtPtr stmt)
      : compilation_error("MALFORMED IR: " + std::to_string(stmt)) {}
  explicit malformed_ir(const std::string& err, StmtPtr stmt)
      : compilation_error(
            "MALFORMED IR: " + err + " - " + std::to_string(stmt)) {}
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
