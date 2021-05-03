#include <benchmark/benchmark.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

using namespace torch::jit::tensorexpr;

namespace {

class ConcatBench : public benchmark::Fixture {
 public:
  void init(const std::vector<std::vector<int>> input_sizes, int concat_dim) {
    input_sizes_ = std::move(input_sizes);
    concat_dim_ = concat_dim;
    inputs_.resize(input_sizes_.size());
    for (size_t i = 0; i < input_sizes_.size(); ++i) {
      inputs_[i] = torch::ones({input_sizes_[i][0], input_sizes_[i][1]});
    }
    output_size_.resize(input_sizes_.front().size());
    for (size_t i = 0; i < output_size_.size(); ++i) {
      if (i == static_cast<size_t>(concat_dim_)) {
        output_size_[i] = 0;
        for (size_t j = 0; j < input_sizes_.size(); ++j) {
          output_size_[i] += input_sizes_[j][i];
        }
      } else {
        output_size_[i] = input_sizes_.front()[i];
      }
    }
    ref_ = at::cat(inputs_, concat_dim_);
    output_ = at::empty_like(ref_);
  }

  void TearDown(benchmark::State& state) override {
    TORCH_CHECK(at::allclose(ref_, output_));
    state.counters["GB/s"] = benchmark::Counter(
        uint64_t(state.iterations()) * 2 * output_.nbytes(),
        benchmark::Counter::kIsRate);
  }

  void runATen(benchmark::State& state) {
    for (auto _ : state) {
      output_ = at::cat(inputs_, concat_dim_);
    }
  }

  void runNNC(benchmark::State& state) {
    KernelScope ks;

    size_t num_inputs = inputs_.size();
    size_t num_dims = 2;

    std::vector<Placeholder> inputs;
    for (size_t i = 0; i < num_inputs; ++i) {
      inputs.emplace_back(Placeholder(
          "input" + std::to_string(i),
          kFloat,
          {input_sizes_[i][0], input_sizes_[i][1]}));
    }

    Tensor* output = Compute(
        "aten_cat",
        {{output_size_[0], "M"}, {output_size_[1], "N"}},
        [&](const VarHandle& m, const VarHandle& n) {
          int d = 0;
          std::vector<int> cumulative_concat_dim_sizes(num_inputs);
          for (size_t i = 0; i < num_inputs; ++i) {
            cumulative_concat_dim_sizes[i] = d;
            d += input_sizes_[i][concat_dim_];
          }
          auto load =
              inputs.back().load(m, n - cumulative_concat_dim_sizes.back());
          for (size_t i = num_inputs - 1; i > 0; --i) {
            load = ifThenElse(
                CompareSelect::make(
                    n, IntImm::make(cumulative_concat_dim_sizes[i]), kLT),
                inputs[i - 1].load(m, n - cumulative_concat_dim_sizes[i - 1]),
                load);
          }
          return load;
        });
    LoopNest nest({output});
    nest.prepareForCodegen();
    Stmt* s = IRSimplifier::simplify(nest.root_stmt());
    std::vector<CodeGen::BufferArg> buf_args(inputs.begin(), inputs.end());
    buf_args.push_back(output);
    LLVMCodeGen cg(s, buf_args);

    std::vector<CodeGen::CallArg> call_args;
    for (auto _ : state) {
      output_ = at::empty_like(ref_);
      call_args.clear();
      for (const auto& inp : inputs_) {
        call_args.push_back(inp.data_ptr<float>());
      }
      call_args.push_back(output_.data_ptr<float>());
      cg.call(call_args);
    }
  }

  void runNNCLoop(benchmark::State& state) {
    KernelScope ks;

    size_t num_inputs = inputs_.size();
    size_t num_dims = 2;

    TORCH_INTERNAL_ASSERT(concat_dim_ == 1);

    auto output_buf = new Buf(
        new Var("aten_cat", kHandle),
        {new IntImm(output_size_[0]), new IntImm(output_size_[1])},
        kFloat);

    std::vector<Placeholder> inputs;
    std::vector<Stmt*> for_stmts(num_inputs);
    int cumulative_input_sizes = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
      inputs.emplace_back(Placeholder(
          "input" + std::to_string(i),
          kFloat,
          {input_sizes_[i][0], input_sizes_[i][1]}));
      std::vector<Var*> for_vars(num_inputs);
      for (size_t d = 0; d < num_dims; ++d) {
        for_vars[d] =
            new Var("i" + std::to_string(i) + "_" + std::to_string(d), kInt);
      }
      auto store = new Store(
          output_buf,
          {for_vars[0],
           new Add(for_vars[1], new IntImm(cumulative_input_sizes))},
          new Load(inputs[i].data(), {for_vars[0], for_vars[1]}));
      auto for_st = new For(
          for_vars[0],
          new IntImm(0),
          new IntImm(input_sizes_[i][0]),
          new For(
              for_vars[1],
              new IntImm(0),
              new IntImm(input_sizes_[i][1]),
              store));
      for_stmts[i] = for_st;
      cumulative_input_sizes += input_sizes_[i][1];
    }
    auto output = new Tensor(output_buf, new Block(for_stmts));

    LoopNest nest({output});
    nest.prepareForCodegen();
    nest.vectorizeInnerLoops();
    Stmt* s = IRSimplifier::simplify(nest.root_stmt());
    std::vector<CodeGen::BufferArg> buf_args(inputs.begin(), inputs.end());
    buf_args.push_back(output);
    LLVMCodeGen cg(s, buf_args);

    std::vector<CodeGen::CallArg> call_args;
    for (auto _ : state) {
      output_ = at::empty_like(ref_);
      call_args.clear();
      for (const auto& inp : inputs_) {
        call_args.push_back(inp.data_ptr<float>());
      }
      call_args.push_back(output_.data_ptr<float>());
      cg.call(call_args);
    }
  }

  std::vector<std::vector<int>> input_sizes_;
  int concat_dim_;
  std::vector<at::Tensor> inputs_;
  std::vector<int> output_size_;
  at::Tensor output_;
  at::Tensor ref_;
};

class Concat2D2Input : public ConcatBench {
 public:
  void SetUp(const benchmark::State& state) override {
    init(
        {{state.range(0), state.range(1)}, {state.range(2), state.range(3)}},
        state.range(4));
  }
};

} // namespace

BENCHMARK_DEFINE_F(Concat2D2Input, ATen)(benchmark::State& state) {
  runATen(state);
}

BENCHMARK_DEFINE_F(Concat2D2Input, NNC)(benchmark::State& state) {
  runNNC(state);
}

BENCHMARK_DEFINE_F(Concat2D2Input, NNCLoop)(benchmark::State& state) {
  runNNCLoop(state);
}

BENCHMARK_REGISTER_F(Concat2D2Input, ATen)
    ->Args({1, 160, 1, 14, 1})
    ->Args({1, 580, 1, 174, 1})
    ->Args({20, 160, 20, 14, 1})
    ->Args({20, 580, 20, 174, 1})
    ->Args({8, 512, 8, 512, 1});

BENCHMARK_REGISTER_F(Concat2D2Input, NNC)
    ->Args({1, 160, 1, 14, 1})
    ->Args({1, 580, 1, 174, 1})
    ->Args({20, 160, 20, 14, 1})
    ->Args({20, 580, 20, 174, 1})
    ->Args({8, 512, 8, 512, 1});

BENCHMARK_REGISTER_F(Concat2D2Input, NNCLoop)
    ->Args({1, 160, 1, 14, 1})
    ->Args({1, 580, 1, 174, 1})
    ->Args({20, 160, 20, 14, 1})
    ->Args({20, 580, 20, 174, 1})
    ->Args({8, 512, 8, 512, 1});

namespace {

class Concat2D3Input : public ConcatBench {
 public:
  void SetUp(const benchmark::State& state) override {
    init(
        {{state.range(0), state.range(1)},
         {state.range(2), state.range(3)},
         {state.range(4), state.range(5)}},
        state.range(6));
  }
};

} // namespace

BENCHMARK_DEFINE_F(Concat2D3Input, ATen)(benchmark::State& state) {
  runATen(state);
}

BENCHMARK_DEFINE_F(Concat2D3Input, NNC)(benchmark::State& state) {
  runNNC(state);
}

BENCHMARK_DEFINE_F(Concat2D3Input, NNCLoop)(benchmark::State& state) {
  runNNCLoop(state);
}

BENCHMARK_REGISTER_F(Concat2D3Input, ATen)->Args({8, 512, 8, 512, 8, 512, 1});

BENCHMARK_REGISTER_F(Concat2D3Input, NNC)->Args({8, 512, 8, 512, 8, 512, 1});

BENCHMARK_REGISTER_F(Concat2D3Input, NNCLoop)
    ->Args({8, 512, 8, 512, 8, 512, 1});

namespace {

class Concat2D7Input : public ConcatBench {
 public:
  void SetUp(const benchmark::State& state) override {
    init(
        {{state.range(0), state.range(1)},
         {state.range(2), state.range(3)},
         {state.range(4), state.range(5)},
         {state.range(6), state.range(7)},
         {state.range(8), state.range(9)},
         {state.range(10), state.range(11)},
         {state.range(12), state.range(13)}},
        state.range(14));
  }
};

} // namespace

BENCHMARK_DEFINE_F(Concat2D7Input, ATen)(benchmark::State& state) {
  runATen(state);
}

BENCHMARK_DEFINE_F(Concat2D7Input, NNC)(benchmark::State& state) {
  runNNC(state);
}

BENCHMARK_DEFINE_F(Concat2D7Input, NNCLoop)(benchmark::State& state) {
  runNNCLoop(state);
}

BENCHMARK_REGISTER_F(Concat2D7Input, ATen)
    ->Args({8, 128, 8, 256, 8, 384, 8, 512, 8, 512, 8, 512, 8, 512, 1});

BENCHMARK_REGISTER_F(Concat2D7Input, NNC)
    ->Args({8, 128, 8, 256, 8, 384, 8, 512, 8, 512, 8, 512, 8, 512, 1});

BENCHMARK_REGISTER_F(Concat2D7Input, NNCLoop)
    ->Args({8, 128, 8, 256, 8, 384, 8, 512, 8, 512, 8, 512, 8, 512, 1});
