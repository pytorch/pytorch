#include <benchmark/benchmark.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

#include <immintrin.h>

namespace te = torch::jit::tensorexpr;

namespace {
class Reduce1D : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) override {
    at::set_num_threads(1);
    torch::manual_seed(0x12345678);
    M = state.range(0);
    A = torch::randn({M});
    B = torch::zeros({});
    ref = torch::sum(A, {0});
  }

  void TearDown(benchmark::State& state) override {
    TORCH_CHECK(at::allclose(B, ref, std::sqrt(A.numel()) * 1e-7));
    state.counters["BYTES"] = benchmark::Counter(
        uint64_t(state.iterations()) * M * sizeof(float),
        benchmark::Counter::kIsRate);
  }

  int M;
  at::Tensor A;
  at::Tensor B;
  at::Tensor ref;
};

} // namespace

BENCHMARK_DEFINE_F(Reduce1D, Torch)(benchmark::State& state) {
  for (auto _ : state) {
    B = torch::sum(A, {0});
  }
}

BENCHMARK_REGISTER_F(Reduce1D, Torch)->Args({1 << 24});

#define VALIDATE(F, A, B) ValidateFunc((F), #F, (A), (B))

template <typename Func>
void ValidateFunc(
    Func func,
    const std::string& func_name,
    at::Tensor& A,
    at::Tensor& B) {
  func(A, B);
  float* pB = B.data_ptr<float>();
  at::Tensor B2 = torch::sum(A, {0});
  float* pB2 = B2.data_ptr<float>();
  int size = A.numel();
  float size_sqrt = std::sqrt(size);
  float natural_noise = size_sqrt * 1e-7;
  if (!torch::allclose(B, B2, natural_noise)) {
    std::ostringstream oss;
    oss << func_name << " failed check: " << std::endl;
    oss << "value: " << B << std::endl;
    ;
    oss << "reference: " << B2 << std::endl;
    oss << "threshold: " << natural_noise << std::endl;
    throw std::runtime_error(oss.str());
  }
}

static void reduce1d_naive(at::Tensor& A, at::Tensor& B) {
  float* pA = A.data_ptr<float>();
  float* pB = B.data_ptr<float>();
  int size = A.numel();
  TORCH_CHECK(B.numel() == 1);
  *pB = 0.;
  for (const auto i : c10::irange(size)) {
    *pB += pA[i];
  }
}

BENCHMARK_DEFINE_F(Reduce1D, Naive)(benchmark::State& state) {
  VALIDATE(reduce1d_naive, A, B);
  for (auto _ : state) {
    reduce1d_naive(A, B);
  }
}

BENCHMARK_REGISTER_F(Reduce1D, Naive)->Args({1 << 24});

static void reduce1d_native_rfactor(at::Tensor& A, at::Tensor& B) {
  float* pA = A.data_ptr<float>();
  float* pB = B.data_ptr<float>();
  int size = A.numel();
  constexpr int kChunkSize = 16;
  TORCH_CHECK(B.numel() == 1);
  TORCH_CHECK(size % kChunkSize == 0);
  *pB = 0.;
  float temp[kChunkSize];
  for (const auto j : c10::irange(kChunkSize)) {
    temp[j] = 0;
  }

  int chunk_count = size / kChunkSize;
  for (const auto i : c10::irange(chunk_count)) {
    for (const auto j : c10::irange(kChunkSize)) {
      temp[j] += pA[i * kChunkSize + j];
    }
  }

  for (const auto j : c10::irange(kChunkSize)) {
    *pB += temp[j];
  }
}

BENCHMARK_DEFINE_F(Reduce1D, NativeRfactor)(benchmark::State& state) {
  VALIDATE(reduce1d_native_rfactor, A, B);
  for (auto _ : state) {
    reduce1d_native_rfactor(A, B);
  }
}

BENCHMARK_REGISTER_F(Reduce1D, NativeRfactor)->Args({1 << 24});

#ifdef USE_AVX2

// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
inline float sum_f32x8(__m256 x) {
  // hiQuad = ( x7, x6, x5, x4 )
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const __m128 loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const __m128 loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const __m128 lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}

static void reduce1d_native_vector(at::Tensor& A, at::Tensor& B) {
  float* pA = A.data_ptr<float>();
  float* pB = B.data_ptr<float>();
  int size = A.numel();
  constexpr int kChunkSize = sizeof(__m256) / sizeof(float);
  TORCH_CHECK(B.numel() == 1);
  TORCH_CHECK(size % kChunkSize == 0);
  *pB = 0.;
  __m256 temp;
  temp = _mm256_setzero_ps();

  int tile_count = size / kChunkSize;
  for (const auto i : c10::irange(tile_count)) {
    __m256 data = _mm256_load_ps(pA + i * kChunkSize);
    temp = _mm256_add_ps(temp, data);
  }

  float result = sum_f32x8(temp);
  *pB = result;
}

BENCHMARK_DEFINE_F(Reduce1D, NativeVector)(benchmark::State& state) {
  VALIDATE(reduce1d_native_vector, A, B);
  for (auto _ : state) {
    reduce1d_native_vector(A, B);
  }
}

BENCHMARK_REGISTER_F(Reduce1D, NativeVector)->Args({1 << 24});

static void reduce1d_native_tiled(at::Tensor& A, at::Tensor& B) {
  static constexpr int kTileSize = 4;
  float* pA = A.data_ptr<float>();
  float* pB = B.data_ptr<float>();
  int size = A.numel();
  constexpr int kChunkSize = sizeof(__m256) / sizeof(float);
  TORCH_CHECK(B.numel() == 1, "Invalid size: ", B.numel(), " != 1");
  TORCH_CHECK(
      size % kChunkSize == 0,
      "Invalid size: ",
      size,
      " % ",
      kChunkSize,
      " ! = 0");
  __m256 t[kTileSize];
  for (const auto j : c10::irange(kTileSize)) {
    t[j] = _mm256_setzero_ps();
  }

  int tile_count = size / kChunkSize / kTileSize;
  for (int i = 0; i < tile_count; i++) {
#pragma unroll
    for (int j = 0; j < kTileSize; j++) {
      float* p = pA + (i * kTileSize + j) * kChunkSize;
      __m256 data = _mm256_loadu_ps(p);
      t[j] = _mm256_add_ps(t[j], data);
    }
  }

  float result = sum_f32x8(t[0]);
  for (const auto j : c10::irange(1, kTileSize)) {
    result += sum_f32x8(t[j]);
  }
  *pB = result;
}

BENCHMARK_DEFINE_F(Reduce1D, NativeTiled)(benchmark::State& state) {
  VALIDATE(reduce1d_native_tiled, A, B);
  for (auto _ : state) {
    reduce1d_native_tiled(A, B);
  }
}

BENCHMARK_REGISTER_F(Reduce1D, NativeTiled)->Args({1 << 24});

#endif // USE_AVX2

BENCHMARK_DEFINE_F(Reduce1D, TeNaive)(benchmark::State& state) {
  int M = A.numel();

  te::BufHandle AP("A", {M}, te::kFloat);
  te::Tensor BT = te::Reduce(
      "reduce_full",
      {1},
      te::Sum(),
      [&](const te::ExprHandle& n, const te::ExprHandle& m) {
        return AP.load(m);
      },
      {M});

  te::LoopNest loop({BT});
  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BT});

  auto func = [&](at::Tensor& A, at::Tensor& B) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>()});
  };

  ValidateFunc(func, "reduce1d_te_naive", A, B);
  for (auto _ : state) {
    func(A, B);
  }
}

BENCHMARK_REGISTER_F(Reduce1D, TeNaive)->Args({1 << 24});

BENCHMARK_DEFINE_F(Reduce1D, TeSplitTail)(benchmark::State& state) {
  int M = A.numel();

  te::BufHandle AP("A", {M}, te::kFloat);
  te::Tensor BT = te::Reduce(
      "reduce_full",
      {1},
      te::Sum(),
      [&](const te::ExprHandle& n, const te::ExprHandle& m) {
        return AP.load(m);
      },
      {M});

  te::LoopNest loop({BT});
  const int kChunkSize = 8;

  {
    auto const& loops = loop.getLoopStmtsFor(BT);
    te::ForPtr m = loops[1];
    loop.splitWithTail(m, kChunkSize);
  }

  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BT});

  auto func = [&](at::Tensor& A, at::Tensor& B) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>()});
  };

  ValidateFunc(func, "reduce1d_te_naive", A, B);
  for (auto _ : state) {
    func(A, B);
  }
}

BENCHMARK_REGISTER_F(Reduce1D, TeSplitTail)->Args({1 << 24});

BENCHMARK_DEFINE_F(Reduce1D, TeSplitMask)(benchmark::State& state) {
  int M = A.numel();

  te::BufHandle AP("A", {M}, te::kFloat);
  te::Tensor BT = te::Reduce(
      "reduce_full",
      {1},
      te::Sum(),
      [&](const te::ExprHandle& n, const te::ExprHandle& m) {
        return AP.load(m);
      },
      {M});

  te::LoopNest loop({BT});
  const int kChunkSize = 8;

  {
    auto const& loops = loop.getLoopStmtsFor(BT);
    te::ForPtr m = loops[1];
    loop.splitWithMask(m, kChunkSize);
  }

  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BT});

  auto func = [&](at::Tensor& A, at::Tensor& B) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>()});
  };

  ValidateFunc(func, "reduce1d_te_naive", A, B);
  for (auto _ : state) {
    func(A, B);
  }
}

BENCHMARK_REGISTER_F(Reduce1D, TeSplitMask)->Args({1 << 24});

BENCHMARK_DEFINE_F(Reduce1D, TeRfactorV1)(benchmark::State& state) {
  int M = A.numel();
  const int kChunkSize = 8;
  TORCH_CHECK(M % kChunkSize == 0);

  te::BufHandle AP("A", {M}, te::kFloat);
  te::Tensor BT = te::Reduce(
      "reduce_full",
      {},
      te::Sum(),
      [&](const te::ExprHandle& m) { return AP.load(m); },
      {M});

  te::LoopNest loop({BT});
  te::BufPtr rfac_buf;

  auto loops = loop.getLoopStmtsFor(BT);
  TORCH_CHECK(loops.size() == 1);
  te::ForPtr mi;
  loop.splitWithMask(loops.at(0), kChunkSize, &mi);
  te::ForPtr mo = loops.at(0);

  loop.reorderAxis(mo, mi);
  loops = loop.getLoopStmtsFor(BT);
  auto bt_body = loop.getAllWritesToBuf(BT.buf())[1];
  TORCH_CHECK(loop.rfactor(bt_body, loops.at(0), &rfac_buf));
  loop.reorderAxis(loops.at(0), loops.at(1));

  loops = loop.getAllInnermostLoopsWritingToBuf(rfac_buf);
  TORCH_CHECK(loops.size() == 2);
  loop.vectorize(loops.at(1));

  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BT});

  auto func = [&](at::Tensor& A, at::Tensor& B) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>()});
  };

  ValidateFunc(func, "reduce1d_te_naive", A, B);
  for (auto _ : state) {
    func(A, B);
  }
}

BENCHMARK_REGISTER_F(Reduce1D, TeRfactorV1)->Args({1 << 24});

BENCHMARK_DEFINE_F(Reduce1D, Op)(benchmark::State& state) {
  const int M = A.numel();
  const int kChunkSize = 8;

  te::BufHandle a("A", {M}, te::kFloat);
  te::Tensor b = te::computeSum(
      {a, te::IntList({0}), false}, {}, {}, at::kFloat, at::kCPU);
  te::LoopNest nest({b});

  auto loops = nest.getLoopStmtsFor(b);
  te::ForPtr mi, mo;
  te::BufPtr rf;
  nest.splitWithMask(loops[0], kChunkSize, &mi);
  loops = nest.reorder({loops[0], mi}, {1, 0});
  nest.rfactor(nest.getLoopBodyFor(b), loops[0], &rf);
  nest.reorderAxis(loops[0], loops[1]);
  for (auto const& loop : nest.getAllInnermostLoopsWritingToBuf(rf)) {
    nest.vectorize(loop);
  }

  nest.prepareForCodegen();
  nest.simplify();
  te::LLVMCodeGen cg(nest.root_stmt(), {a, b});

  for (auto _ : state) {
    cg.call({A.data_ptr<float>(), B.data_ptr<float>()});
  }
}
BENCHMARK_REGISTER_F(Reduce1D, Op)->Args({1 << 24});

class Reduce2DCol : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) override {
    at::set_num_threads(1);
    torch::manual_seed(0x12345678);
    M = state.range(0);
    N = state.range(1);
    A = torch::randn({M, N});
    ref = torch::sum(A, {0});
    B = torch::zeros_like(ref);
  }

  void TearDown(benchmark::State& state) override {
    TORCH_CHECK(at::allclose(B, ref, std::sqrt(A.numel()) * 1e-5));
    state.counters["BYTES"] = benchmark::Counter(
        uint64_t(state.iterations()) * (A.nbytes() + B.nbytes()),
        benchmark::Counter::kIsRate);
  }

  int M;
  int N;
  at::Tensor A;
  at::Tensor B;
  at::Tensor ref;
};

BENCHMARK_DEFINE_F(Reduce2DCol, Torch)(benchmark::State& state) {
  for (auto _ : state) {
    B = torch::sum(A, {0});
  }
}
BENCHMARK_REGISTER_F(Reduce2DCol, Torch)
    ->Args({1 << 3, 1 << 21})
    ->Args({1 << 6, 1 << 18})
    ->Args({1 << 12, 1 << 12});

BENCHMARK_DEFINE_F(Reduce2DCol, OpSchedule)(benchmark::State& state) {
  constexpr int kCacheSize = 1 << 12;
  te::BufHandle a("A", {M, N}, te::kFloat);
  te::Tensor b = te::computeSum(
      {a, te::IntList({0}), false}, {N}, {1}, at::kFloat, at::kCPU);
  te::LoopNest nest({b});

  auto sch = state.range(2);
  if (sch == 0) {
  } else if (sch == 1) {
    auto loops = nest.getLoopStmtsFor(b);
    nest.reorderAxis(loops[0], loops[1]);
  } else if (sch == 2) {
    auto loops = nest.getLoopStmtsFor(b);
    nest.splitWithTail(loops[0], kCacheSize);
    loops = nest.getLoopStmtsFor(b);
    nest.reorderAxis(loops[1], loops[2]);
  } else if (sch == 3) {
    auto loops = nest.getLoopStmtsFor(b);
    nest.splitWithTail(loops[1], 8);
    loops = nest.getLoopStmtsFor(b);
    nest.reorderAxis(loops[0], loops[1]);
  }

  nest.prepareForCodegen();
  nest.simplify();
  te::LLVMCodeGen cg(nest.root_stmt(), {a, b});
  for (auto _ : state) {
    cg.call({A.data_ptr<float>(), B.data_ptr<float>()});
  }
}
BENCHMARK_REGISTER_F(Reduce2DCol, OpSchedule)
    ->Apply( // CustomArgs);
        [](benchmark::internal::Benchmark* b) {
          for (auto sch : {0, 1, 2, 3}) {
            for (auto rows : {3, 6, 12}) {
              auto cols = 24 - rows;
              b->Args({1 << rows, 1 << cols, sch});
            }
          }
        });

class Reduce2DRow : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) override {
    at::set_num_threads(1);
    torch::manual_seed(0x12345678);
    M = state.range(0);
    N = state.range(1);
    A = torch::randn({M, N});
    ref = torch::sum(A, {1});
    B = torch::zeros_like(ref);
  }

  void TearDown(benchmark::State& state) override {
    TORCH_CHECK(at::allclose(B, ref, std::sqrt(A.numel()) * 1e-4));
    state.counters["BYTES"] = benchmark::Counter(
        uint64_t(state.iterations()) * (A.nbytes() + B.nbytes()),
        benchmark::Counter::kIsRate);
  }

  int M;
  int N;
  at::Tensor A;
  at::Tensor B;
  at::Tensor ref;
};

BENCHMARK_DEFINE_F(Reduce2DRow, Torch)(benchmark::State& state) {
  for (auto _ : state) {
    B = torch::sum(A, {1});
  }
}
BENCHMARK_REGISTER_F(Reduce2DRow, Torch)
    ->Args({1 << 3, 1 << 21})
    ->Args({1 << 6, 1 << 18})
    ->Args({1 << 12, 1 << 12})
    ->Args({1 << 18, 1 << 6});

BENCHMARK_DEFINE_F(Reduce2DRow, Hand)(benchmark::State& state) {
  auto a = A.data_ptr<float>();
  auto b = B.data_ptr<float>();
  constexpr int Mb = 4;
  constexpr int Nb = 4;
  auto fn = [&] {
    for (int m_outer = 0; m_outer < M; m_outer += Mb) {
      float bregs[Mb][Nb] = {0.0f};
      for (int n_outer = 0; n_outer < N; n_outer += Nb) {
        for (const auto m_inner : c10::irange(Mb)) {
          for (const auto n_inner : c10::irange(Nb)) {
            bregs[m_inner][n_inner] +=
                a[(m_outer + m_inner) * N + n_outer + n_inner];
          }
        }
      }
      for (const auto m_inner : c10::irange(Mb)) {
        b[m_outer + m_inner] = 0.f;
        for (const auto n_inner : c10::irange(Nb)) {
          b[m_outer + m_inner] += bregs[m_inner][n_inner];
        }
      }
    }
  };
  for (auto _ : state) {
    fn();
  }
}
BENCHMARK_REGISTER_F(Reduce2DRow, Hand)->Args({1 << 18, 1 << 6});

BENCHMARK_DEFINE_F(Reduce2DRow, OpSchedule)(benchmark::State& state) {
  constexpr int kChunkSize = 8;
  te::BufHandle a("A", {M, N}, te::kFloat);
  te::Tensor b = te::computeSum(
      {a, te::IntList({1}), false}, {M}, {1}, at::kFloat, at::kCPU);
  te::LoopNest nest({b});

  auto sch = state.range(2);
  if (sch == 1) {
    auto loops = nest.getLoopStmtsFor(b);
    te::ForPtr mi, mo;
    te::BufPtr rf;
    nest.splitWithMask(loops[1], kChunkSize, &mi);
    loops = nest.reorder({loops[1], mi}, {1, 0});
    TORCH_CHECK(nest.rfactor(nest.getLoopBodyFor(b), loops[0], &rf));
    nest.reorderAxis(loops[0], loops[1]);
    for (auto const& loop : nest.getAllInnermostLoopsWritingToBuf(rf)) {
      nest.vectorize(loop);
    }
  } else if (sch == 2) {
    auto loops = nest.getLoopStmtsFor(b);
    nest.splitWithMask(loops[1], 8);
    nest.splitWithMask(loops[0], 4);
    loops = nest.getLoopStmtsFor(b);
    nest.reorderAxis(loops[1], loops[2]);
  } else if (sch == 3) {
    auto loops = nest.getLoopStmtsFor(b);
    te::ForPtr mi, mo;
    te::BufPtr rf;
    nest.splitWithMask(loops[1], kChunkSize, &mi);
    loops = nest.reorder({loops[1], mi}, {1, 0});
    TORCH_CHECK(nest.rfactor(nest.getLoopBodyFor(b), loops[0], &rf));
    nest.reorderAxis(loops[0], loops[1]);
    te::LoopNest::compressBuffer(rf, nest.root_stmt());
    for (auto const& loop : nest.getAllInnermostLoopsWritingToBuf(rf)) {
      nest.vectorize(loop);
    }
  }

  nest.prepareForCodegen();
  nest.simplify();
  te::LLVMCodeGen cg(nest.root_stmt(), {a, b});

  for (auto _ : state) {
    cg.call({A.data_ptr<float>(), B.data_ptr<float>()});
  }
}
BENCHMARK_REGISTER_F(Reduce2DRow, OpSchedule)
    ->Apply( // CustomArgs);
        [](benchmark::internal::Benchmark* b) {
          for (auto sch : {0, 1, 2, 3}) {
            for (auto rows : {3, 6, 12, 18}) {
              auto cols = 24 - rows;
              b->Args({1 << rows, 1 << cols, sch});
            }
          }
        });
