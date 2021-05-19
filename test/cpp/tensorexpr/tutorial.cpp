// *** Tensor Expressions ***
//
// This tutorial covers basics of NNC's tensor expressions, shows basic APIs to
// work with them, and outlines how they are used in the overall TorchScript
// compilation pipeline. This doc is permanently a "work in progress" since NNC
// is under active development and things change fast.
//
// This Tutorial's code is compiled in the standard pytorch build, and the
// executable can be found in `build/bin/tutorial_tensorexpr`.
//
// *** What is NNC ***
//
// NNC stands for Neural Net Compiler. It is a component of TorchScript JIT
// and it performs on-the-fly code generation for kernels, which are often a
// combination of multiple aten (torch) operators.
//
// When the JIT interpreter executes a torchscript model, it automatically
// extracts subgraphs from the torchscript IR graph for which specialized code
// can be JIT generated. This usually improves performance as the 'combined'
// kernel created from the subgraph could avoid unnecessary memory traffic that
// is unavoidable when the subgraph is interpreted as-is, operator by operator.
// This optimization is often referred to as 'fusion'. Relatedly, the process of
// finding and extracting subgraphs suitable for NNC code generation is done by
// a JIT pass called 'fuser'.
//
// *** What is TE ***
//
// TE stands for Tensor Expressions. TE is a commonly used approach for
// compiling kernels performing tensor (~matrix) computation. The idea behind it
// is that operators are represented as a mathematical formula describing what
// computation they do (as TEs) and then the TE engine can perform mathematical
// simplification and other optimizations using those formulas and eventually
// generate executable code that would produce the same results as the original
// sequence of operators, but more efficiently.
//
// NNC's design and implementation of TE was heavily inspired by Halide and TVM
// projects.
#include <iostream>
#include <string>

#include <torch/csrc/jit/tensorexpr/eval.h>
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

  std::cout << "*** Structure of tensor expressions ***" << std::endl;
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
    ExprHandle load = Load::make(A.dtype(), A, {i, j});
    std::cout << "Tensor expression: " << *load.node() << std::endl;
    // Prints: Tensor expression: A[i, j]
  }

  std::cout << "*** Tensors, Functions, and Placeholders ***" << std::endl;
  {
    // A tensor computation is represented by Tensor class objects and
    // consists of the following pieces:
    //   - domain, which is specified by a Buf expression
    //   - a tensor statement, specified by a Stmt object, that computation to
    //   be performed in this domain

    // Let's start with defining a domain. We do this by creating a Buf object.

    // First, let's specify the sizes:
    std::vector<const Expr*> dims = {
        new IntImm(64), new IntImm(32)}; // IntImm stands for Integer Immediate
    // and represents an integer constant

    // Now we can create a Buf object by providing a name, dimensions, and a
    // data type of the elements:
    const Buf* buf = new Buf("X", dims, kInt);

    // Next we need to spefify the computation. We can do that by either
    // constructing a complete tensor statement for it (statements are
    // examined in details in subsequent section), or by using a convenience
    // method where we could specify axis and an element expression for the
    // computation. In the latter case a corresponding statement would be
    // constructed automatically.

    // Let's define two variables, i and j - they will be axis in our
    // computation.
    const Var* i = new Var("i", kInt);
    const Var* j = new Var("j", kInt);
    std::vector<const Var*> args = {i, j};

    // Now we can define the body of the tensor computation using these
    // variables. What this means is that values in our tensor are:
    //   X[i, j] = i * j
    Expr* body = new Mul(i, j);

    // Finally, we pass all these pieces together to Tensor constructor:
    Tensor* X = new Tensor(buf, args, body);
    std::cout << "Tensor computation: " << *X << std::endl;
    // Prints:
    // Tensor computation: Tensor X[64, 32]:
    // for (int i = 0; i < 64; i++) {
    //   for (int j = 0; j < 32; j++) {
    //     X[i, j] = i * j;
    //   }
    // }

    // TODO: Add an example of constructing a Tensor with a complete Stmt.

    // Similarly to how we provide a more convenient way of using handles for
    // constructing Exprs, Tensors also have a more convenient API for
    // construction. It is based on Compute API, which takes a name,
    // dimensions, and a lambda specifying the computation body:
    Tensor* Z = Compute(
        "Z",
        {{64, "i"}, {32, "j"}},
        [](const VarHandle& i, const VarHandle& j) { return i / j; });
    std::cout << "Tensor computation: " << *Z << std::endl;
    // Prints:
    // Tensor computation: Tensor Z[64, 32]:
    // for (int i = 0; i < 64; i++) {
    //   for (int j = 0; j < 32; j++) {
    //     Z[i, j] = i / j;
    //   }
    // }

    // Tensors might access other tensors and external placeholders in their
    // expressions. It can be done like so:
    Placeholder P("P", kInt, {64, 32});
    Tensor* R = Compute(
        "R",
        {{64, "i"}, {32, "j"}},
        [&](const VarHandle& i, const VarHandle& j) {
          return Z->load(i, j) * P.load(i, j);
        });
    std::cout << "Tensor computation: " << *R << std::endl;
    // Prints:
    // Tensor computation: Tensor R[64, 32]:
    // for (int i = 0; i < 64; i++) {
    //   for (int j = 0; j < 32; j++) {
    //     R[i, j] = (Z(i, j)) * (P[i, j]);
    //   }
    // }

    // Placeholders could be thought of as external tensors, i.e. tensors for
    // which we don't have the element expression. In other words, for `Tensor`
    // we know an expression specifying how its elements can be computed (a
    // mathematical formula). For external tensors, or placeholders, we don't
    // have such an expression. They need to be considered as coming to us as
    // inputs from outside - we can only load data from them.
    //
    // TODO: Show how reductions are represented and constructed
  }

  std::cout << "*** Loopnests and Statements ***" << std::endl;
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
          return sigmoid(X->load(i, j));
        });
    std::cout << "Tensor computation X: " << *X
              << "Tensor computation Y: " << *Y << std::endl;
    // Prints:
    // Tensor computation X: Tensor X[64, 32]:
    // for (int i = 0; i < 64; i++) {
    //   for (int j = 0; j < 32; j++) {
    //     X[i, j] = (A[i, j]) + (B[i, j]);
    //   }
    // }

    // Tensor computation Y: Tensor Y[64, 32]:
    // for (int i = 0; i < 64; i++) {
    //   for (int j = 0; j < 32; j++) {
    //     Y[i, j] = sigmoid(X(i, j));
    //   }
    // }

    // Creating a loop nest is as quite simple, we just need to specify a list
    // of all and a list of output tensors:
    // NOLINTNEXTLINE(bugprone-argument-comment)
    LoopNest loopnest(/*outputs=*/{Y}, /*all=*/{X, Y});

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
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    For* j_inner;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    For* j_tail;
    int split_factor = 9;
    loopnest.splitWithTail(
        loops[1], // loops[0] is the outer loop, loops[1] is inner
        split_factor,
        &j_inner, // further transformations
        &j_tail);
    // loops[1] will become the outer loop, j_outer, after splitWithTail.
    std::cout << *loopnest.root_stmt() << std::endl;
    // Prints:
    // {
    //   for (int i = 0; i < 64; i++) {
    //     for (int j_outer = 0; j_outer < (32 - 0) / 9; j_outer++) {
    //       for (int j_inner = 0; j_inner < 9; j_inner++) {
    //         Y[i, j_outer * 9 + j_inner] = sigmoid((A[i, j_outer * 9 + ...
    //       }
    //     }
    //     for (int j_tail = 0; j_tail < (32 - 0) % 9; j_tail++) {
    //       Y[i, j_tail + ((32 - 0) / 9) * 9] = sigmoid((A[i, j_tail + ...
    //     }
    //   }
    // }

    // TODO: List all available transformations
    // TODO: Show how statements can be constructed manually
  }

  std::cout << "*** Codegen ***" << std::endl;
  {
    // An ultimate goal of tensor expressions is to be provide a mechanism to
    // execute a given computation in the fastest possible way. So far we've
    // looked at how we could describe what computation we're interested in, but
    // we haven't looked at how to actually execute it. So far all we've been
    // dealing with was just symbols with no actual data associated, in this
    // section we would look at how we can bridge that gap.

    // Let's start by constructing a simple computation for us to work with:
    Placeholder A("A", kInt, {64, 32});
    Placeholder B("B", kInt, {64, 32});
    Tensor* X = Compute(
        "X",
        {{64, "i"}, {32, "j"}},
        [&](const VarHandle& i, const VarHandle& j) {
          return A.load(i, j) + B.load(i, j);
        });

    // And let's lower it to a loop nest, as we did in the previous section:
    LoopNest loopnest({X});
    std::cout << *loopnest.root_stmt() << std::endl;
    // Prints:
    // {
    //   for (int i = 0; i < 64; i++) {
    //     for (int j = 0; j < 32; j++) {
    //       X[i, j] = (A[i, j]) + (B[i, j]);
    //     }
    //   }

    // Now imagine that we have two actual tensors 64x32 that we want sum
    // together, how do we pass those tensors to the computation and how do we
    // carry it out?
    //
    // Codegen object is aimed at providing exactly that functionality. Codegen
    // is an abstract class and concrete codegens are derived from it.
    // Currently, we have three codegens:
    //  1) Simple Evaluator,
    //  2) LLVM Codegen for CPU,
    //  3) CUDA Codegen.
    // In this example we will be using Simple Evaluator, since it's available
    // everywhere.

    // To create a codegen, we need to provide the statement - it specifies the
    // computation we want to perform - and a list of placeholders and tensors
    // used in the computation. The latter part is crucial since that's the only
    // way the codegen could use to correlate symbols in the statement to actual
    // data arrays that we will be passing when we will actually be performing
    // the computation.
    //
    // Let's create a Simple IR Evaluator codegen for our computation:
    SimpleIREvaluator ir_eval(loopnest.root_stmt(), {A, B, X});

    // We are using the simplest codegen and in it almost no work is done at the
    // construction step. Real codegens such as CUDA and LLVM perform
    // compilation during that stage so that when we're about to run the
    // computation everything is ready.

    // Let's now create some inputs and run our computation with them:
    std::vector<int> data_A(64 * 32, 3); // This will be the input A
    std::vector<int> data_B(64 * 32, 5); // This will be the input B
    std::vector<int> data_X(64 * 32, 0); // This will be used for the result

    // Now let's invoke our codegen to perform the computation on our data. We
    // need to provide as many arguments as how many placeholders and tensors we
    // passed at the codegen construction time. A position in these lists would
    // define how real data arrays from the latter call (these arguments are
    // referred to as 'CallArg's in our codebase) correspond to symbols
    // (placeholders and tensors) used in the tensor expressions we constructed
    // (these are referred to as 'BufferArg').
    // Thus, we will provide three arguments: data_A, data_B, and data_X. data_A
    // contains data for the placeholder A, data_B - for the placeholder B, and
    // data_X would be used for contents of tensor X.
    ir_eval(data_A, data_B, data_X);

    // Let's print one of the elements from each array to verify that the
    // computation did happen:
    std::cout << "A[10] = " << data_A[10] << std::endl
              << "B[10] = " << data_B[10] << std::endl
              << "X[10] = A[10] + B[10] = " << data_X[10] << std::endl;
    // Prints:
    // A[10] = 3
    // B[10] = 5
    // X[10] = A[10] + B[10] = 8
  }

  // TODO: Show how TorchScript IR is translated to TE
  return 0;
}
