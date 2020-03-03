#pragma once

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

static std::unordered_map<UnaryOpType, std::string> unary_op_type_inline_op_string_map; //Defined in type.cpp

struct Fusion;

struct Statement;

struct Val;
struct Expr;

struct UnaryOp;
struct BinaryOp;

struct Tensor;
struct TensorDomain;
struct TensorView;

struct TensorContiguity;

struct IterDomain;

struct Float;
struct Int;
struct Add;

struct TORCH_API Printer{
std::ostream& os;
bool print_inline_ = false;
public:

Printer(std::ostream& _os):os(_os){}

virtual void print(const Fusion* const);
virtual void print(const Fusion& f){print(&f);}

virtual void print(const Statement* const);

virtual void print(const Val* const);
virtual void print(const Expr* const);

virtual void print(const Tensor* const);
virtual void print(const TensorDomain* const);
virtual void print(const TensorView* const);
virtual void print(const IterDomain* const);
virtual void print(const TensorContiguity* const);

virtual void print(const Float* const);
virtual void print(const Int* const);

virtual void print(const UnaryOp* const);
virtual void print(const BinaryOp* const);

virtual void print(const Split* const);
virtual void print(const Merge* const);
virtual void print(const Reorder* const);

void print_inline(const Statement* const stmt){
  bool prev = print_inline_;
  print_inline_ = true;
  print(stmt);
  print_inline_ = prev;
}

};

TORCH_API std::ostream& operator<< (std::ostream& os, const Statement* const stmt);
TORCH_API std::ostream& operator<< (std::ostream& os, const Fusion* const f);
TORCH_API std::ostream& operator<< (std::ostream& os, const Fusion& f);

} // namespace fuser
} // namespace jit
} // namespace torch
