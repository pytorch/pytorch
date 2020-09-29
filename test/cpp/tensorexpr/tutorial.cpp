// TODO: Add an intro
#include <iostream>
#include <string>

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

using namespace torch::jit::tensorexpr;

int main(int argc, char* argv[]) {
  // TODO: Describe what KernelScope is
  KernelScope kernel_scope;
  {
    // *** Structure of tensor expressions ***
    //
    // A tensor expression is a tree of expressions. Each expression has a type,
    // and that type defines what sub-expressions it the current expression has.
    // For instance, an expression of type 'Mul' would have a type 'kMul' and
    // two subexpressions: LHS and RHS. Each of these two sub-expressions could
    // also be a 'Mul' or some other expression.
    //
    // Let's construct a simple TE:
    Expr* lhs = new IntImm(5);
    Expr* rhs = new Var("x", kInt);
    Expr* mul = new Mul(lhs, rhs);
    std::cout << "Tensor expression: " << *mul << std::endl;
    // Prints: Tensor expression: 5 * x

    // Here we created an expression representing a 5*x computation, where x is
    // an int variable.

    // Another, probably a more convenient, way to construct tensor expressions
    // is to use so called expression handles (as opposed to raw expressions
    // like we did in the previous example). Expression handles overload common
    // operations and allow us to express the same semantics in a more natural
    // way:
    ExprHandle l = 1;
    ExprHandle r = Var::make("x", kInt);
    ExprHandle m = l * r;
    std::cout << "Tensor expression: " << *m.node() << std::endl;
    // Prints: Tensor expression: 1 * x

    // In a similar fashion we could construct arbitrarily complex expressions
    // using mathematical and logical operations, casts between various data
    // types, and a bunch of intrinsics.
    ExprHandle a = Var::make("a", kInt);
    ExprHandle b = Var::make("b", kFloat);
    ExprHandle c = Var::make("c", kFloat);
    ExprHandle x = ExprHandle(5) * a + b / (sigmoid(c) - 3.0f);
    std::cout << "Tensor expression: " << *x.node() << std::endl;
    // Prints: Tensor expression: float(5 * a) + b / ((sigmoid(c)) - 3.f)

    // An ultimate purpose of tensor expressions is to optimize tensor
    // computations, and in order to represent accesses to tensors data, there
    // is a special kind of expression - a load.
    // To construct a load we need two pieces: the base and the indices. The
    // base of a load is a Buf expression, which could be thought of as a
    // placeholder similar to Var, but with dimensions info.
    //
    // Let's construct a simple load:
    BufHandle A("A", {ExprHandle(64), ExprHandle(32)}, kInt);
    ExprHandle i = Var::make("i", kInt), j = Var::make("j", kInt);
    ExprHandle load = Load::make(A.dtype(), A, {i, j}, /* mask= */ 1);
    std::cout << "Tensor expression: " << *load.node() << std::endl;
    // Prints: Tensor expression: A[i, j]
  }
  {
    // *** Tensors, Functions, and Placeholders ***
    //
    // A tensor computation is represented by objects of Tensor class and
    // consists of the following pieces:
    //   - domain, which is specified by a Buf expression
    //   - an expression (or several expressions if we want to perform several
    //   independent computations over the same domain) for its elements, as a
    //   function of indices
    //
    // We use Function objects to represent this. Let's build one:
    std::vector<const Expr*> dims = {new IntImm(64), new IntImm(32)};

    // Function arguments:
    const Var* i = new Var("i", kInt);
    const Var* j = new Var("j", kInt);
    std::vector<const Var*> args = {i, j};
    // Element expressions:
    Expr* func_body1 = new Mul(i, j);
    Expr* func_body2 = new Add(i, j);

    Function* func =
        new Function({"X", "Y"}, dims, args, {func_body1, func_body2});
    std::cout << "Tensor function: " << *func << std::endl;
    // Prints:
    // Tensor function: Function F(i[64], j[32]) {
    //   X = i * j
    //   Y = i + j
    // }

    // A Tensor refers to an individual computation defined by a Function. For
    // instance, we could create a following tensor given the function above:
    int output_idx = 0; // Used to index the computation
    Tensor* X = new Tensor(func, output_idx);
    std::cout << "Tensor computation: " << *X << std::endl;
    // Prints: Tensor computation: Tensor X(i[64], j[32]) = i * j

    // Similarly to how we provide a more convenient way of using handles for
    // constructing Exprs, Tensors also have a more convenient API for
    // construction. It is based on Compute functions, which take a name:
    // dimensions, and a lambda specifying the computation body:
    Tensor* Z = Compute(
        "Z",
        {{64, "i"}, {32, "j"}},
        [](const VarHandle& i, const VarHandle& j) { return i / j; });
    std::cout << "Tensor computation: " << *Z << std::endl;
    // Prints: Tensor computation: Tensor Z(i[64], j[32]) = i / j

    // TODO: Show how reductions are represented and constructed
  }
  // TODO: Show how TorchScript IR is translated to TE
  // TODO: Describe statements
  // TODO: Describe codegen
  // TODO: Describe Loop Nests and loop transformations
  return 0;
}
