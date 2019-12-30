// #pragma once

// #include <torch/csrc/jit/fuser/common/ir.h>
// #include <torch/csrc/jit/fuser/common/ir_visitor.h>

// #include <ostream>

// namespace torch {
// namespace jit {
// namespace fuser {

// struct IRPrinter : public IRVisitor {

//   IRPrinter() = delete;
//   IRPrinter(std::ostream& _os)
//   : os{_os}
//   , indent_count{0}  { }

//   // Copy constructor and copy assignment operator
//   IRPrinter(const IRPrinter& other) = default;
//   IRPrinter& operator=(const IRPrinter& other) = delete;

//   ~IRPrinter() { }

//   virtual void visit(const Add* v) override {};
//   virtual void visit(const Sub* v) override {};
//   virtual void visit(const Mul* v) override {};
//   virtual void visit(const Div* v) override {};
//   virtual void visit(const IntImm* v) override {};
//   virtual void visit(const FloatImm* v) override {};
//   virtual void visit(const Cast* v) override {};
//   virtual void visit(const Variable* v) override {};
//   virtual void visit(const For* v) override {};
//   virtual void visit(const Block* v) override {};
//   virtual void visit(const EmptyExpr* v) override {};

//   void print(Expr expr) {
//     expr.accept(this);
//   }

// private:
//   void indent();
//   std::ostream& os;
//   int indent_count;
// };

// std::ostream& operator<<(std::ostream& stream, const Expr&);

// } // namespace fuser
// } // namespace jit
// } // namespace torch
