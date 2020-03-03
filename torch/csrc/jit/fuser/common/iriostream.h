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

void print(const Fusion* const);
void print(const Fusion& f){print(&f);}

void print(const Statement* const);

void print(const Val* const);
void print(const Expr* const);

void print(const Tensor* const);
void print(const TensorDomain* const);
void print(const TensorView* const);
void print(const IterDomain* const);
void print(const TensorContiguity* const);

void print(const Float* const);
void print(const Int* const);

void print(const UnaryOp* const);
void print(const BinaryOp* const);

void print(const Split* const);
void print(const Merge* const);
void print(const Reorder* const);

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
