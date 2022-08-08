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

#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

using namespace torch::jit::tensorexpr;

#ifdef TORCH_ENABLE_LLVM

// Helper function to print a snippet from a big multi-line string
static void printLinesToFrom(const std::string& input_str, int from, int to);

#endif

int main(int argc, char* argv[]) {
  std::cout << "*** Structure of tensor expressions and statements ***"
            << std::endl;
  {
    // A tensor expression is a tree of expressions. Each expression has a type,
    // and that type defines what sub-expressions the current expression has.
    // For instance, an expression of type 'Mul' would have a type 'kMul' and
    // two subexpressions: LHS and RHS. Each of these two sub-expressions could
    // also be a 'Mul' or some other expression.
    //
    // Let's construct a simple TE:
    ExprPtr lhs = alloc<IntImm>(5);
    ExprPtr rhs = alloc<Var>("x", kInt);
    ExprPtr mul = alloc<Mul>(lhs, rhs);
    std::cout << "Tensor expression: " << *mul << std::endl;
    // Prints: Tensor expression: 5 * x

    // Here we created an expression representing a 5*x computation, where x is
    // an int variable.

    // Another, probably a more convenient, way to construct tensor expressions
    // is to use so called expression handles (as opposed to raw expressions
    // like we did in the previous example). Expression handles overload common
    // operations and allow us to express the same semantics in a more natural
    // way:
    ExprHandle l = 5;
    ExprHandle r = Var::make("x", kInt);
    ExprHandle m = l * r;
    std::cout << "Tensor expression: " << *m.node() << std::endl;
    // Prints: Tensor expression: 5 * x

    // Converting from handles to raw expressions and back is easy:
    ExprHandle handle = Var::make("x", kInt);
    ExprPtr raw_expr_from_handle = handle.node();
    ExprPtr raw_expr = alloc<Var>("x", kInt);
    ExprHandle handle_from_raw_expr = ExprHandle(raw_expr);

    // We could construct arbitrarily complex expressions using mathematical
    // and logical operations, casts between various data types, and a bunch of
    // intrinsics.
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
    BufHandle A("A", {64, 32}, kInt);
    VarPtr i_var = alloc<Var>("i", kInt), j_var = alloc<Var>("j", kInt);
    ExprHandle i(i_var), j(j_var);
    ExprHandle load = Load::make(A.dtype(), A, {i, j});
    std::cout << "Tensor expression: " << *load.node() << std::endl;
    // Prints: Tensor expression: A[i, j]

    // Tensor Expressions constitute Tensor Statements, which are used to
    // represent computation of a given operator or a group of operators from a
    // fusion group.
    //
    // There are three main kinds of tensor statements:
    //  - block
    //  - store
    //  - loop
    //
    // A Store represents a store to a single element of a tensor (or to a
    // group of elements if it's a vectorized store). Store statements,
    // similarly to Load expressions, have a base and indices, but on top of
    // that they also include a value - an expression representing what needs
    // to be stored at the given memory location. Let's create a Store stmt:
    StmtPtr store_a = Store::make(A, {i, j}, i + j);
    std::cout << "Store statement: " << *store_a << std::endl;
    // Prints: Store statement: A[i, j] = i + j;

    // An operator fills the entire tensor, not just a single element, and to
    // represent this we need to use For stmt: let's wrap our store stmt with
    // two nested loops to represent that variables i and j need to iterate
    // over some ranges.
    ForPtr loop_j_a = For::make(VarHandle(j_var), 0, 32, store_a);
    ForPtr loop_i_a = For::make(VarHandle(i_var), 0, 64, loop_j_a);

    std::cout << "Nested for loops: " << std::endl << *loop_i_a << std::endl;
    // Prints:
    // Nested for loops:
    // for (const auto i : c10::irange(64)) {
    //   for (const auto j : c10::irange(32)) {
    //     A[i, j] = i + j;
    //   }
    // }

    // A Block statement is used when we need a sequence of other statements.
    // E.g. if a fusion group contains several operators, we initially define
    // separate loopnest for each of them and put them all into a common block:
    BufHandle B("B", {64, 32}, kInt);
    StmtPtr store_b = Store::make(B, {i, j}, A.load(i, j));
    ForPtr loop_j_b = For::make(VarHandle(j_var), 0, 32, store_b);
    ForPtr loop_i_b = For::make(VarHandle(i_var), 0, 64, loop_j_b);

    BlockPtr block = Block::make({loop_i_a, loop_i_b});
    std::cout << "Compound Block statement: " << std::endl
              << *block << std::endl;
    // Prints:
    // Compound Block statement:
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       A[i, j] = i + j;
    //     }
    //   }
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       B[i, j] = A[i, j];
    //     }
    //   }
    // }

    // Manually constructing nested loops and blocks to represent a computation
    // might be laborious, and instead we can use a 'Compute' API. This API
    // requires us to specify dimensions and a lambda to compute a single
    // element of the resulting tensor and returns a `Tensor` structure. This
    // structure is simply a pair of a buffer that was created to represent the
    // result of the computation (BufPtr) and a statement representing the
    // computation itself (StmtPtr).
    Tensor C =
        Compute("C", {64, 32}, [&](const VarHandle& i, const VarHandle& j) {
          return i * j;
        });
    std::cout << "Stmt produced by 'Compute' API: " << std::endl
              << *C.stmt() << std::endl;
    // Prints:
    // Stmt produced by 'Compute' API:
    // for (const auto i : c10::irange(64)) {
    //   for (const auto j : c10::irange(32)) {
    //     C[i, j] = i * j;
    //   }
    // }

    // To construct statements to represent computations with reductions, we
    // can use a 'Reduce' API - it is similar to 'Compute' but takes a couple
    // of extra arguments defining how to perform the reduction. Let's define a
    // simple 2D sum of C using that:
    Tensor D = Reduce(
        "D",
        {},
        Sum(),
        [&](const VarHandle& i, const VarHandle& j) { return C.load(i, j); },
        {64, 32});
    std::cout << "Stmt produced by 'Reduce' API: " << std::endl
              << *D.stmt() << std::endl;
  }

  std::cout << "*** Loopnests transformations ***" << std::endl;
  {
    // When a statement for the computation is generated, we might want to
    // apply some optimizations to it. These transformations allow us to end up
    // with a statement producing the same results, but more efficiently.
    //
    // Let's look at a couple of transformations that are used in NNC. We will
    // begin with constructing a Block statement like we did before.

    Tensor C =
        Compute("C", {64, 32}, [&](const VarHandle& i, const VarHandle& j) {
          return i * (j + 1);
        });
    BufHandle c_buf(C.buf());
    Tensor D =
        Compute("D", {64, 32}, [&](const VarHandle& i, const VarHandle& j) {
          return c_buf.load(i, j) - i;
        });
    StmtPtr block = Block::make({C.stmt(), D.stmt()});
    std::cout << "Stmt produced by 'Compute' API: " << std::endl
              << *block << std::endl;
    // Prints:
    // Stmt produced by 'Compute' API:
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       C[i, j] = i * (j + 1);
    //     }
    //   }
    //   for (const auto i_1 : c10::irange(64)) {
    //     for (const auto j_1 : c10::irange(32)) {
    //       D[i_1, j_1] = (C[i_1, j_1]) - i_1;
    //     }
    //   }
    // }

    // One transformation we can apply to this computation is inlining: i.e.
    // taking the expression that defines values of C and substituting a load
    // from C with it.
    // To do that, we first need to create a special object called LoopNest -
    // all transformations are methods of this class. To create a loopnest we
    // need to provide a list of output buffers and the root statement:
    LoopNest nest(block, {D.buf()});

    // We can always retrieve the Stmt back from LoopNest:
    std::cout << "LoopNest root stmt: " << std::endl
              << *nest.root_stmt() << std::endl;
    // Prints:
    // LoopNest root stmt:
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       C[i, j] = i * (j + 1);
    //     }
    //   }
    //   for (const auto i_1 : c10::irange(64)) {
    //     for (const auto j_1 : c10::irange(32)) {
    //       D[i_1, j_1] = (C[i_1, j_1]) - i_1;
    //     }
    //   }
    // }

    // Now we can apply the inlining transformation:
    nest.computeInline(C.buf());
    std::cout << "Stmt after inlining:" << std::endl
              << *nest.root_stmt() << std::endl;
    // Prints:
    // Stmt after inlining:
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       D[i, j] = i * (j + 1) - i;
    //     }
    //   }
    // }

    // We can also apply algebraic simplification to a statement:
    StmtPtr simplified = IRSimplifier::simplify(nest.root_stmt());
    std::cout << "Stmt after simplification:" << std::endl
              << *simplified << std::endl;
    // Prints:
    // Stmt after simplification:
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       D[i, j] = i * j;
    //     }
    //   }
    // }

    // Many loopnest transformations are stateless and can be applied without
    // creating a LoopNest object. In fact, we plan to make all transformations
    // stateless.
    // splitWithTail is one such transformation: it splits an iteration space
    // of a given loop into two with a given factor.
    ForPtr outer_loop = to<For>(to<Block>(simplified)->stmts().front());
    LoopNest::splitWithTail(outer_loop, 13);
    // Call simplifier once more to fold some arithmetic.
    simplified = IRSimplifier::simplify(simplified);
    std::cout << "Stmt after splitWithTail:" << std::endl
              << *simplified << std::endl;
    // Prints:
    // Stmt after splitWithTail:
    // {
    //   for (const auto i_outer : c10::irange(4)) {
    //     for (const auto i_inner : c10::irange(13)) {
    //       for (const auto j : c10::irange(32)) {
    //         D[i_inner + 13 * i_outer, j] = i_inner * j + 13 * (i_outer * j);
    //       }
    //     }
    //   }
    //   for (const auto i_tail : c10::irange(12)) {
    //     for (const auto j : c10::irange(32)) {
    //       D[i_tail + 52, j] = i_tail * j + 52 * j;
    //     }
    //   }
    // }

    // NNC supports a wide range of loop nest transformations, which we are not
    // listing here. Please refer to documentation in
    // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/tensorexpr/loopnest.h
    // for more details.
  }

  std::cout << "*** Codegen ***" << std::endl;
  {
    // An ultimate goal of tensor expressions is to be provide a mechanism to
    // execute a given computation in the fastest possible way. So far we've
    // looked at how we could describe what computation we're interested in, but
    // we haven't looked at how to actually execute it.
    //
    // All we've been dealing with was just symbols with no actual data
    // associated, in this section we would look at how we can bridge that gap.

    // Let's start by constructing a simple computation for us to work with:
    BufHandle A("A", {64, 32}, kInt);
    BufHandle B("B", {64, 32}, kInt);
    Tensor X =
        Compute("X", {64, 32}, [&](const VarHandle& i, const VarHandle& j) {
          return A.load(i, j) + B.load(i, j);
        });

    // And let's lower it to a loop nest, as we did in the previous section. We
    // can pass Tensor object directly:
    LoopNest loopnest({X});
    std::cout << *loopnest.root_stmt() << std::endl;
    // Prints:
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
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

  std::cout << "*** Lowering TorchScript IR to TensorExpr IR ***" << std::endl;
  {
    // This section requires a LLVM-enabled PyTorch build, so we have to use a
    // guard:
#ifdef TORCH_ENABLE_LLVM

    // Often we would like to convert a TorchScript IR to TE rather than
    // construct TE IR from scratch.  NNC provides an API to perform such
    // lowering: it takes a TorchScript graph and returns an object that can be
    // used to invoke the generated kernel.
    // This API is currently used by the TorchScript JIT fuser and can also be
    // used ahead of time to pre-compile parts of a model.
    //
    // To get familiar with this API let's first start with defining a simple
    // TorchScript graph:
    const auto graph_string = R"IR(
        graph(%A : Float(5, 3, strides=[3, 1], device=cpu),
              %B : Float(5, 3, strides=[3, 1], device=cpu)):
          %AB : Float(5, 3, strides=[3, 1]) = aten::mul(%A, %B)
          %one : int = prim::Constant[value=1]()
          %AAB : Float(5, 3, strides=[3, 1]) = aten::mul(%A, %AB)
          %AAB_plus_B: Float(5, 3, strides=[3, 1]) = aten::add(%AAB, %B, %one)
          return (%AAB_plus_B))IR";
    auto graph = std::make_shared<torch::jit::Graph>();
    parseIR(graph_string, &*graph);

    // This graph defines a simple computation of A*A*B + B where A and B are
    // input 5x3 tensors.

    // To lower this TorchScript graph to TE, we just need to create a
    // TensorExprKernel object. In its constructor it constructs the
    // corresponding TE IR and compiles it for the given backend (in this
    // example for CPU using LLVM compiler).
    TensorExprKernel kernel(graph);

    // We can retrieve the generated TE stmt from the kernel object:
    StmtPtr kernel_stmt = kernel.getCodeGenStmt();
    std::cout << "TE Stmt constructed from TorchScript: " << std::endl
              << *kernel_stmt << std::endl;
    // Prints:
    // TE Stmt constructed from TorchScript:
    // {
    //   for (const auto v : c10::irange(5)) {
    //     for (const auto _tail_tail : c10::irange(3)) {
    //       aten_add[_tail_tail + 3 * v] = (tA[_tail_tail + 3 * v]) *
    //       ((tA[_tail_tail + 3 * v]) * (tB[_tail_tail + 3 * v])) +
    //       (tB[_tail_tail + 3 * v]);
    //     }
    //   }
    // }

    // We can also examine generated LLVM IR and assembly code:
    std::cout << "Generated LLVM IR: " << std::endl;
    auto ir_str = kernel.getCodeText("ir");
    printLinesToFrom(ir_str, 15, 20);
    // Prints:
    // Generated LLVM IR:
    //   %9 = bitcast float* %2 to <8 x float>*
    //   %10 = load <8 x float>, <8 x float>* %9 ...
    //   %11 = bitcast float* %5 to <8 x float>*
    //   %12 = load <8 x float>, <8 x float>* %11 ...
    //   %13 = fmul <8 x float> %10, %12
    //   %14 = fmul <8 x float> %10, %13

    std::cout << "Generated assembly: " << std::endl;
    auto asm_str = kernel.getCodeText("asm");
    printLinesToFrom(asm_str, 10, 15);
    // Prints:
    // Generated assembly:
    //         vmulps  %ymm1, %ymm0, %ymm2
    //         vfmadd213ps     %ymm1, %ymm0, %ymm2
    //         vmovups %ymm2, (%rax)
    //         vmovss  32(%rcx), %xmm0
    //         vmovss  32(%rdx), %xmm1
    //         vmulss  %xmm1, %xmm0, %xmm2

    // We can also execute the generated kernel:
    auto A =
        at::ones({5, 3}, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) *
        2.0;
    auto B =
        at::ones({5, 3}, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) *
        3.0;
    std::vector<at::Tensor> inputs = {A, B};
    std::vector<torch::IValue> stack = torch::fmap<torch::IValue>(inputs);
    kernel.run(stack);
    auto R = stack[0].toTensor();

    // Let's print one of the elements from the result tensor to verify that the
    // computation did happen and was correct:
    std::cout << "R[2][2] = " << R[2][2] << std::endl;
    // Prints:
    // R[2][2] = 15
    // [ CPUFloatType{} ]
#endif
  }
  return 0;
}

void printLinesToFrom(const std::string& input_str, int from, int to) {
  std::istringstream f(input_str);
  std::string s;
  int idx = 0;
  while (getline(f, s)) {
    if (idx > from) {
      std::cout << s << "\n";
    }
    if (idx++ > to) {
      break;
    }
  }
}
