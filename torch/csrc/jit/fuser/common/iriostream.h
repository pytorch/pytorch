#pragma once

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

struct Statement;

struct Val;
struct Expr;

struct Tensor;
struct TensorDomain;
struct TensorView;

struct TensorContiguity;

struct IterDomain;

struct Float;
struct Int;
struct Add;

TORCH_API std::ostream& operator<<(std::ostream& os, const Fusion* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Statement* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Val* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const Expr* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Tensor* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const TensorDomain* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const TensorView* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const TensorContiguity* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const IterDomain* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Float* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const Int* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const UnaryOp* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const BinaryOp* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Split* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const Merge* const);
TORCH_API std::ostream& operator<<(std::ostream& os, const Reorder* const);

TORCH_API std::ostream& operator<<(std::ostream& os, const Fusion& f);

} // namespace fuser
} // namespace jit
} // namespace torch
