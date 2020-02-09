#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

struct Statement;
struct Tensor;
struct Float;
struct Int;
struct Add;
struct Val;
struct Expr;

struct FusionGurad;
struct Fusion;


struct TORCH_API IterVisitor {
  virtual ~IterVisitor() = default;

public:
  IterVisitor() = default;

  IterVisitor(const IterVisitor& other) = default;
  IterVisitor& operator=(const IterVisitor& other) = default;

  IterVisitor(IterVisitor&& other) = default;
  IterVisitor& operator=(IterVisitor&& other) = default;

  std::vector<const Statement*> next(const Statement* const stmt);
  std::vector<const Statement*> next(const Expr* const expr);
  std::vector<const Statement*> next(const Val* const v);

  virtual void handle(const Statement* const stmt);
  virtual void handle(const Expr* const expr);
  virtual void handle(const Val* const val);
  virtual void handle(const Float* const f);
  virtual void handle(const Tensor* const t);
  virtual void handle(const Int* const i);
  virtual void handle(const Add* const add);

  void traverse(const Fusion* const _fusion,  bool from_outputs_only, bool breadth_first);

};

}}} //torch::jit::fuser