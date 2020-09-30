// TODO: Add an intro
#include <iostream>
#include <string>

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

using namespace torch::jit::tensorexpr;

int main(int argc, char* argv[]) {
  // Memory management for tensor expressions is currently done with memory
  // arenas. That is, whenever an object is created it registers itself in an
  // arena and the object is kept alive as long as the arena is alive. When the
  // arena gets destructed, it deletes all objects registered in it.
  //
  // The easiest way to set up a memory arena is to use `KernelScope` class - it
  // is a resource guard that creates a new arena on construction and restores
  // the previously set arena on destruction.
  //
  // We will create a kernel scope here, and thus we'll set up a mem arena for
  // the entire tutorial.
  KernelScope kernel_scope;

  // *** Structure of tensor expressions ***
  {
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

  // *** Tensors, Functions, and Placeholders ***
  {
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

    // Tensors might access other tensors and external placeholders in their
    // expressions. It can be done like so:
    Placeholder P("P", kFloat, {64, 32});
    Tensor* R = Compute(
        "R",
        {{64, "i"}, {32, "j"}},
        [&](const VarHandle& i, const VarHandle& j) {
          return Z->call(i, j) * P.load(i, j);
        });
    std::cout << "Tensor computation: " << *R << std::endl;
    // Prints: Tensor computation: Tensor R(i[64], j[32]) = Z(i, j) * P[i, j]

    // Placeholders could be thought of as external tensors, i.e. tensors for
    // which we don't have the element expression.
    // Also note that we use 'call' to construct an access to an element of a
    // Tensor and we use 'load' for accessing elements of an external tensor
    // through its Placeholder. This is an implementation detail and could be
    // changed in future.

    // TODO: Show how reductions are represented and constructed
  }

  // *** Loopnests and Statements ***
  {
    // Creating a tensor expression is the first step to generate an executable
    // code for it. A next step is to represent it as a loop nest and apply
    // various loop transformations in order to get an optimal implementation.
    // In Halide's or TVM's terms the first step was to define the algorithm of
    // computation (what to compute?) and now we are getting to the schedule of
    // the computation (how to compute?).
    //
    // Let's create a simple tensor expression and construct a loop nest for it.
    Placeholder A("A", kFloat, {64, 32});
    Placeholder B("B", kFloat, {64, 32});
    Tensor* X = Compute(
        "X",
        {{64, "i"}, {32, "j"}},
        [&](const VarHandle& i, const VarHandle& j) {
          return A.load(i, j) + B.load(i, j);
        });
    Tensor* Y = Compute(
        "Y",
        {{64, "i"}, {32, "j"}},
        [&](const VarHandle& i, const VarHandle& j) {
          return sigmoid(X->call(i, j));
        });
    std::cout << "Tensor computation X: " << *X
              << "Tensor computation Y: " << *Y << std::endl;
    // Prints:
    // Tensor computation X: Tensor X(i[64], j[32]) = (A[i, j]) + (B[i, j])
    // Tensor computation Y: Tensor Y(i[64], j[32]) = sigmoid(X(i, j))

    // Creating a loop nest is as quite simple, we just need to specify what are
    // the output tensors in our computation and LoopNest object will
    // automatically pull all tensor dependencies:
    LoopNest loopnest({Y});

    // An IR used in LoopNest is based on tensor statements, represented by
    // `Stmt` class. Statements are used to specify the loop nest structure, and
    // to take a sneak peek at them, let's print out what we got right after
    // creating our LoopNest object:
    std::cout << *loopnest.root_stmt() << std::endl;
    // Prints:
    // {
    //   for (int i = 0; i < 64; i++) {
    //     for (int j = 0; j < 32; j++) {
    //       X[i, j] = (A[i, j]) + (B[i, j]);
    //     }
    //   }
    //   for (int i_1 = 0; i_1 < 64; i_1++) {
    //     for (int j_1 = 0; j_1 < 32; j_1++) {
    //       Y[i_1, j_1] = sigmoid(X(i_1, j_1));
    //     }
    //   }
    // }

    // To introduce statements let's first look at their three main types (in
    // fact, there are more than 3 types, but the other types would be easy to
    // understand once the overall structure is clear):
    //  1) Block
    //  2) For
    //  3) Store
    //
    // A `Block` statement is simply a list of other statements.
    // A `For` is a statement representing one axis of computation. It contains
    // an index variable (Var), boundaries of the axis (start and end - both are
    // `Expr`s), and a `Block` statement body.
    // A `Store` represents an assignment to a tensor element. It contains a Buf
    // representing the target tensor, a list of expressions for indices of the
    // element, and the value to be stored, which is an arbitrary expression.

    // Once we've constructed the loop nest, we can apply various tranformations
    // to it. To begin with, let's inline computation of X into computation of Y
    // and see what happens to our statements.
    loopnest.computeInline(loopnest.getLoopBodyFor(X));
    std::cout << *loopnest.root_stmt() << std::endl;
    // Prints:
    // {
    //   for (int i = 0; i < 64; i++) {
    //     for (int j = 0; j < 32; j++) {
    //       Y[i, j] = sigmoid((A[i, j]) + (B[i, j]));
    //     }
    //   }
    // }
    //
    // As you can see, the first two loops have disappeared and the expression
    // for X[i,j] has been inserted into the Y[i,j] computation.

    // Loop transformations can be composed, so we can do something else with
    // our loop nest now. Let's split the inner loop with a factor of 9, for
    // instance.
    std::vector<For*> loops = loopnest.getLoopStmtsFor(Y);
    For* j_outer;
    For* j_inner;
    For* j_tail;
    loopnest.splitWithTail(
        loops[1], // loops[0] is the outer loop, loops[1] is inner
        9,
        &j_outer, // These are handles that we would be using for
        &j_inner, // further transformations
        &j_tail);
    std::cout << *loopnest.root_stmt() << std::endl;
    // Prints:
    // {
    //   for (int i = 0; i < 64; i++) {
    //     for (int j_outer = 0; j_outer < (32 - 0) / 9; j_outer++) {
    //       for (int j_inner = 0; j_inner < 9; j_inner++) {
    //         Y[i, j_outer * 9 + j_inner] = sigmoid((A[i, j_outer * 9 +
    //         j_inner]) + (B[i, j_outer * 9 + j_inner]));
    //       }
    //     }
    //     for (int j_tail = 0; j_tail < (32 - 0) % 9; j_tail++) {
    //       Y[i, j_tail + ((32 - 0) / 9) * 9] = sigmoid((A[i, j_tail + ((32 -
    //       0) / 9) * 9]) + (B[i, j_tail + ((32 - 0) / 9) * 9]));
    //     }
    //   }
    // }

    // TODO: List all available transformations
    // TODO: Show how statements can be constructed manually
  }

  // TODO: Describe codegen
  // TODO: Show how TorchScript IR is translated to TE
  return 0;
}
