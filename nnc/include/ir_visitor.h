#ifndef NNC_INCLUDE_IR_VISITOR_H_INCLUDED_
#define NNC_INCLUDE_IR_VISITOR_H_INCLUDED_

namespace nnc {

class Add;
class Sub;
class Mul;
class Div;
class IntImm;
class FloatImm;
class Cast;
class Variable;
class Let;

class IRVisitor {
 public:
  virtual void visit(const Add* v);
  virtual void visit(const Sub* v);
  virtual void visit(const Mul* v);
  virtual void visit(const Div* v);
  virtual void visit(const IntImm* v);
  virtual void visit(const FloatImm* v);
  virtual void visit(const Cast* v);
  virtual void visit(const Variable* v);
  virtual void visit(const Let* v);
};

}  // namespace nnc

#endif  // NNC_INCLUDE_IR_VISITOR_H_INCLUDED_
