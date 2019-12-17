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
class Ramp;
class Load;
class For;
class Block;
class Store;
class Broadcast;

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
  virtual void visit(const Ramp* v);
  virtual void visit(const Load* v);
  virtual void visit(const For* v);
  virtual void visit(const Block* v);
  virtual void visit(const Store* v);
  virtual void visit(const Broadcast* v);
};

}  // namespace nnc

#endif  // NNC_INCLUDE_IR_VISITOR_H_INCLUDED_
