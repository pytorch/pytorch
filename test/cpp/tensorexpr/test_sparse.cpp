#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

namespace te = torch::jit::tensorexpr;
namespace F = torch::nn::functional;

TEST(Sparse, SparseDenseCSRMV) {
  te::KernelScope ks;

  int M = 1, K = 16;
  int N = K * 2;

  auto data = torch::linspace(1.f, M * K, M * K).reshape({M, K});
  auto weight_data = torch::linspace(1.f, N, N).reshape({N});
  auto weight_indices = torch::linspace(0, K - 1, N, at::TensorOptions().dtype(at::kInt));
  auto weight_indptr = torch::arange(0, N + 2, 2, at::TensorOptions().dtype(torch::kInt));

  te::Placeholder data_t("data", te::kFloat, {M, K});
  te::Placeholder weight_data_t("weight_data", te::kFloat, {N});
  te::Placeholder weight_indices_t("weight_indices", te::kInt, {N});
  te::Placeholder weight_indptr_t("weight_indices", te::kInt, {K + 1});

  /*
    def f(i, row):
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem]
        weight_val = data[i, weight_indices[elem]]
        return tvm.sum(a_val * weight_val, axis=elem_idx)
    return tvm.compute(oshape, f)
  */

/*
  te::Tensor* output_t = te::Compute(
      "output",
      {{M, "M"}, {K, "K"}},
      [&](const te::VarHandle& i, const te::VarHandle& row) {
        auto row_start = weight_indptr_t.load(row);
        auto row_end = weight_indptr_t.load(row + 1);
        auto row_elems = row_end - row_start;

        return te::Sum()(
            {row_elems},
            [&](const VarHandle& elem_idx) {
              auto elem = row_start + elem_idx;
              auto a_val = weight_data.load(elem);
              auto weight_val = data.load(i, weight_indices.load(elem));
              return a_val * weight_val;
            }
        );
      });
*/

  /*
  te::Tensor* output_t = te::Reduce(
      "output",
      {{M, "M"}, {K, "K"}},
      te::Sum(),
      [&](const te::VarHandle& i, const te::VarHandle& row, const te::VarHandle& row_elems) {
        auto row_start = weight_indptr_t.load(row);
        auto row_end = weight_indptr_t.load(row + 1);
        auto row_elems = row_end - row_start;

        auto elem = row_start + elem_idx;
        auto a_val = data.load(i, weight_indices.load(elem));
        auto weight_val = weight_data.load(elem);
        return a_val * weight_val;
      },
      {{row_elems, "row_elems"}}
  );
  */

  te::Tensor* output_t = te::Compute(
      "output",
      {{M, "M"}, {K, "K"}},
      [&](const te::VarHandle& i, const te::VarHandle& row) {
        auto row_start = weight_indptr_t.load(row);
        auto row_end = weight_indptr_t.load(row + 1);
        auto row_elems = row_end - row_start;

        auto sum = te::SumX(
            {{row_elems, "row_elems"}},
            [&](const std::vector<te::VarHandle>& args) {
              auto elem_idx = args[0];
              auto elem = row_start + elem_idx;
              auto weight_val = weight_data_t.load(elem);
              auto a_val = data_t.load(i, weight_indices_t.load(elem));
              return a_val * weight_val;
            }
        );
        return te::ExprHandle(sum);
      });
  std::clog << "tensor stmt:\n" << *output_t->stmt() << "\n";

  te::LoopNest nest({output_t});
  std::clog << "loop nest pre cg:\n" << *nest.root_stmt() << "\n";

  nest.prepareForCodegen();
  te::Stmt* s = nest.root_stmt();
  std::clog << "loop nest post cg:\n" << *nest.root_stmt() << "\n";
  s = te::IRSimplifier::simplify(s);

  at::Tensor output = torch::empty({M, K});
  te::LLVMCodeGen cg(s, {data_t, weight_data_t, weight_indices_t, weight_indptr_t, output_t});
  cg.call({
      data.data_ptr<float>(),
        weight_data.data_ptr<float>(),
        weight_indices.data_ptr<int32_t>(),
        weight_indptr.data_ptr<int32_t>(),
        output.data_ptr<float>(),
    });
  std::clog << data << "\n"
            << weight_data << "\n"
            << weight_indices << "\n"
            << weight_indptr << "\n"
            << output << "\n";
}

}
}
