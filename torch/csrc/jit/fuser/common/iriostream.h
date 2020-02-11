#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

struct Statement;

struct Val;
struct Expr;

struct Tensor;
struct Float;
struct Int;
struct Add;

TORCH_API std::ostream& operator<<(std::ostream& os, const Fusion* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Statement* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Val* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const Expr* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Tensor* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const Float* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const Int* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const UnaryOp* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const BinaryOp* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Fusion& f);

} // namespace fuser
} // namespace jit
} // namespace torch
