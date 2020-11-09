
#include <torch/csrc/jit/passes/fox.h>

#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {

void foxPass() {
    KernelScope kernel_scope;

    Placeholder A("A", kFloat, {64, 32});
    Placeholder B("B", kFloat, {64, 32});
    Tensor* X = Compute(
        "X",
        {{64, "i"}, {32, "j"}},
        [&](const VarHandle& i, const VarHandle& j) {
            return A.load(i, j) + B.load(i, j);
        });

    LoopNest loopnest({X});
    std::vector<For*> loops = loopnest.getLoopStmtsFor(X);
    loopnest.vectorizeInnerLoops();
    Stmt* stmt = loopnest.root_stmt();
    // Arithmetic Simplification.
    stmt = IRSimplifier::simplify(stmt);
    std::cout << *stmt << std::endl;

    LLVMCodeGen ir_eval(stmt, {A, B, X});

    std::vector<float> data_A(64 * 32, 3);
    std::vector<float> data_B(64 * 32, 5);
    std::vector<float> data_X(64 * 32, 0);

    std::vector<void*> buf_addrs = {
        &data_A[0],
        &data_B[0],
        &data_X[0]
    };
    ir_eval.value<float>(buf_addrs);

    std::cout << "A[10] = " << data_A[10] << std::endl
              << "B[10] = " << data_B[10] << std::endl
              << "X[10] = A[10] + B[10] = " << data_X[10] << std::endl;
}

tensorexpr::KernelArena* enterNewKernelScope() {
    auto* old_arena = KernelArena::GetCurrentKernelArena();
    KernelArena::SetCurrentKernelArena(new KernelArena);
    return old_arena;
}

void exitKernelScope(tensorexpr::KernelArena *orig) {
    delete KernelArena::GetCurrentKernelArena();
    KernelArena::SetCurrentKernelArena(orig);
}

}}  // namespace torch::jit
