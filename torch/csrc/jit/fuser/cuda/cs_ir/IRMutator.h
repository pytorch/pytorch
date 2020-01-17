#pragma once
#include <memory>

namespace Fuser{

template<typename T>
class ExprNode;

struct Expr;

class IntImm;
class Add;
class Sub;
class Mul;
class Div;
class Mod;
class LT;
class Set;
class Variable;
class Tensor;
class TensorAccessor;
class For;
class If;
class Attr;
class Thread;
class Block;

class IRMutator {
public:
    IRMutator();
    virtual ~IRMutator();

    virtual Expr mutate(const Expr &expr);
    
protected:
    // ExprNode<> and StmtNode<> are allowed to call visit (to implement mutate_expr/mutate_stmt())
    template<typename T>
    friend struct ExprNode;
    
    virtual Expr visit(const IntImm *);
    virtual Expr visit(const Add *);
    virtual Expr visit(const Sub *);
    virtual Expr visit(const Mul *);
    virtual Expr visit(const Div *);
    virtual Expr visit(const Mod *);
    virtual Expr visit(const LT *);
    virtual Expr visit(const Set *);
    virtual Expr visit(const Variable *);
    virtual Expr visit(const Tensor *);
    virtual Expr visit(const TensorAccessor *);
    virtual Expr visit(const For *);
    virtual Expr visit(const If *);
    virtual Expr visit(const Attr *);
    virtual Expr visit(const Thread *);
    virtual Expr visit(const Block *);
};

}
