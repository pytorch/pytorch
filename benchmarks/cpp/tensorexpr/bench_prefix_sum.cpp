#include <benchmark/benchmark.h>
#include "ATen/Functions.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

#include <immintrin.h>

using namespace torch::jit::tensorexpr;

namespace {
#ifdef __AVX2__

#define _mm256_slli_si1(x)                                                   \
  _mm256_blend_epi32(                                                        \
      _mm256_permutevar8x32_ps(x, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7)), \
      _mm256_setzero_si256(),                                                \
      1)
#define _mm256_slli_si2(x)                                                   \
  _mm256_blend_epi32(                                                        \
      _mm256_permutevar8x32_ps(x, _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 6)), \
      _mm256_setzero_si256(),                                                \
      3)
#define _mm256_slli_si4(x)                                                   \
  _mm256_blend_epi32(                                                        \
      _mm256_permutevar8x32_ps(x, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4)), \
      _mm256_setzero_si256(),                                                \
      15)

__m256 PrefixSum(__m256 x) {
  x = _mm256_add_ps(x, _mm256_slli_si1(x));
  x = _mm256_add_ps(x, _mm256_slli_si2(x));
  x = _mm256_add_ps(x, _mm256_slli_si4(x));
  return x; // local prefix sums
}

__m256i PrefixSumInt(__m256i x) {
  x = _mm256_add_epi32(x, _mm256_slli_si1(x));
  x = _mm256_add_epi32(x, _mm256_slli_si2(x));
  x = _mm256_add_epi32(x, _mm256_slli_si4(x));
  return x; // local prefix sums
}

// Util function to log the given value. Not used during benchmarking.
template <class T>
inline void Log(const __m256i& value) {
  const size_t n = sizeof(__m256i) / sizeof(T);
  T buffer[n];
  _mm256_storeu_si256((__m256i*)buffer, value);
  for (int i = 0; i < n; i++)
    std::cout << buffer[n - i - 1] << " ";
  std::cout << std::endl;
}
#endif

#ifdef __AVX512F__

#define _mm512_slli_si512(x, k) \
  _mm512_alignr_epi32(x, _mm512_setzero_si512(), 16 - k)

__m512 PrefixSum(__m512 x) {
  x = _mm512_add_ps(x, _mm512_slli_si512(x, 1));
  x = _mm512_add_ps(x, _mm512_slli_si512(x, 2));
  x = _mm512_add_ps(x, _mm512_slli_si512(x, 4));
  x = _mm512_add_ps(x, _mm512_slli_si512(x, 8));
  return x; // local prefix sums
}

__m512i PrefixSumInt(__m512i x) {
  x = _mm512_add_epi32(x, _mm512_slli_si512(x, 1));
  x = _mm512_add_epi32(x, _mm512_slli_si512(x, 2));
  x = _mm512_add_epi32(x, _mm512_slli_si512(x, 4));
  x = _mm512_add_epi32(x, _mm512_slli_si512(x, 8));
  return x; // local prefix sums
}

template <int index>
float _mm512_extract_f32(__m512 target) {
  return _mm512_cvtss_f32(_mm512_alignr_epi32(target, target, index));
}

// extract the last i32 from target
int _mm512_extract_epi32(__m512i target) {
  __m256i x = _mm512_extracti32x8_epi32(target, 1);
  return _mm256_extract_epi32(x, 7);
}

void PrefixSum(float* output_data, float* input_data, size_t input_size) {
  float carry = 0.0f;
  for (int i = 0; i < input_size / 16; i++) {
    __m512 x = _mm512_loadu_ps(input_data + i * 16);
    x = PrefixSum(x);
    x = _mm512_add_ps(x, _mm512_set1_ps(carry));
    carry = _mm512_extract_f32<15>(x);
    _mm512_storeu_ps((__m512*)(output_data + i * 16), x);
  }
}

void PrefixSum(int* output_data, int* input_data, size_t input_size) {
  int carry = 0;
  for (int i = 0; i < input_size / 16; i++) {
    __m512i x = _mm512_loadu_epi32(input_data + i * 16);
    x = PrefixSumInt(x);
    x = _mm512_add_epi32(x, _mm512_set1_epi32(carry));
    carry = _mm512_extract_epi32(x);
    _mm512_storeu_epi32((__m512i*)(output_data + i * 16), x);
  }
}
#endif

// PrefixSum: the same as inclusive scan
class PrefixSumBench : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) override {
    input_size_ = state.range(0);
    input_ = torch::rand(input_size_);
    ref_ = prefixSum(input_);

    // no type promotion. Default is int->long.
    input_int_ = torch::randint(1000, {input_size_}, at::kInt);
    ref_int_ = at::cumsum(input_int_, 0, at::kInt);
  }

  void TearDown(benchmark::State& state) override {
    if (output_.numel() > 0) {
      if (output_.numel() == ref_.numel()) {
        TORCH_CHECK(at::allclose(ref_, output_, 1e-3, 1e-3));
      }
      state.counters["GB/s"] = benchmark::Counter(
          uint64_t(state.iterations()) * 2 * output_.nbytes(),
          benchmark::Counter::kIsRate);
    } else {
      if (output_int_.numel() == ref_int_.numel()) {
        TORCH_CHECK(ref_int_.equal(output_int_));
      }
      state.counters["GB/s"] = benchmark::Counter(
          uint64_t(state.iterations()) * 2 * output_int_.nbytes(),
          benchmark::Counter::kIsRate);
    }
  }

  at::Tensor prefixSum(const at::Tensor& inp) {
    return at::cumsum(inp, 0);
  }

  void runATen(benchmark::State& state) {
    output_ = prefixSum(input_);
    for (auto _ : state) {
      at::cumsum_out(output_, input_, 0);
    }
  }

  void runLocal(benchmark::State& state) {
    output_ = at::empty_like(ref_);
    for (auto _ : state) {
      auto input_data = input_.data_ptr<float>();
      auto output_data = output_.data_ptr<float>();
      float sum = 0.0f;
      for (int i = 0; i < input_size_; ++i) {
        sum = sum + input_data[i];
        output_data[i] = sum;
      }
    }
  }

  // no type promotion
  void runLocalInt(benchmark::State& state) {
    output_int_ = at::empty_like(input_int_);
    for (auto _ : state) {
      auto input_data = input_int_.data_ptr<int>();
      auto output_data = output_int_.data_ptr<int>();
      int sum = 0;
      for (int i = 0; i < input_size_; ++i) {
        sum = sum + input_data[i];
        output_data[i] = sum;
      }
    }
  }

  void runNNC(benchmark::State& state) {
    BufHandle input("input", {input_size_}, kFloat);
    BufHandle output("output", {input_size_}, kFloat);
    BufHandle s("s", {1}, kFloat);
    VarHandle i("i", kInt);
    auto allocS = Allocate::make(s);
    auto initS = Store::make(s, {0}, 0.0f);
    auto accumS = Store::make(
        s, {0}, Add::make(Load::make(s, {0}), Load::make(input, {i})));
    auto store = Store::make(output, {i}, Load::make(s, {0}));
    auto forI = For::make(i, 0, input_size_, Block::make({accumS, store}));
    auto freeS = Free::make(s);
    auto par = Block::make({allocS, initS, forI, freeS});
    LoopNest nest(par, {output.node()});

    std::vector<CodeGen::BufferArg> buf_args;
    buf_args.emplace_back(input);
    buf_args.emplace_back(output);
    LLVMCodeGen cg(nest.root_stmt(), buf_args);

    std::vector<CodeGen::CallArg> call_args;
    output_ = at::empty_like(ref_);
    for (auto _ : state) {
      call_args.clear();
      call_args.emplace_back(input_.data_ptr<float>());
      call_args.emplace_back(output_.data_ptr<float>());
      cg.call(call_args);
    }
  }

#ifdef __AVX2__
  void runLocalAVX2(benchmark::State& state) {
    output_ = at::empty_like(ref_);
    for (auto _ : state) {
      float* input_data = input_.data_ptr<float>();
      float* output_data = output_.data_ptr<float>();

      float carry = 0.0f;
      for (int i = 0; i < input_size_ / 8; i++) {
        __m256 x = _mm256_loadu_ps(input_data + i * 8);
        x = PrefixSum(x);
        x = _mm256_add_ps(x, _mm256_set1_ps(carry));
        (reinterpret_cast<__m256*>(output_data))[i] = x;
        carry = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(
            x, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7)));
      }
    }
  }

  void runLocalIntAVX2(benchmark::State& state) {
    output_int_ = at::empty_like(input_int_);
    for (auto _ : state) {
      auto input_data = input_int_.data_ptr<int>();
      auto output_data = output_int_.data_ptr<int>();

      int carry = 0;
      for (size_t i = 0; i < input_size_ / 8; i++) {
        __m256i x = _mm256_loadu_si256((__m256i*)(input_data + i * 8));
        x = PrefixSumInt(x);
        x = _mm256_add_epi32(x, _mm256_set1_epi32(carry));
        _mm256_storeu_si256((__m256i*)(output_data + i * 8), x);
        carry = _mm256_extract_epi32(x, 7);
      }
    }
  }
#endif

#ifdef __AVX512F__
  void runLocalAVX512(benchmark::State& state) {
    output_ = at::empty_like(ref_);
    for (auto _ : state) {
      auto input_data = input_.data_ptr<float>();
      auto output_data = output_.data_ptr<float>();
      PrefixSum(output_data, input_data, input_size_);
    }
  }

  void runLocalIntAVX512(benchmark::State& state) {
    output_int_ = at::empty_like(input_int_);
    for (auto _ : state) {
      auto input_data = input_int_.data_ptr<int>();
      auto output_data = output_int_.data_ptr<int>();
      PrefixSum(output_data, input_data, input_size_);
    }
  }

  void runExclusiveScanAVX512(benchmark::State& state) {
    output_ = at::empty({input_size_ + 1}, at::kFloat);
    for (auto _ : state) {
      auto input_data = input_.data_ptr<float>();
      auto output_data = output_.data_ptr<float>();
      output_data[0] = 0.0f;
      PrefixSum(output_data + 1, input_data, input_size_);
    }
  }

  void runExclusiveScanIntAVX512(benchmark::State& state) {
    output_int_ = at::empty({input_size_ + 1}, at::kInt);
    for (auto _ : state) {
      auto input_data = input_int_.data_ptr<int>();
      auto output_data = output_int_.data_ptr<int>();
      output_data[0] = 0;
      PrefixSum(output_data + 1, input_data, input_size_);
    }
  }

#endif

 private:
  int input_size_;
  at::Tensor input_;
  at::Tensor output_;
  at::Tensor ref_;
  at::Tensor input_int_;
  at::Tensor output_int_;
  at::Tensor ref_int_; // no type promotion
};

} // namespace

BENCHMARK_DEFINE_F(PrefixSumBench, ATen)(benchmark::State& state) {
  runATen(state);
}

BENCHMARK_DEFINE_F(PrefixSumBench, Local)(benchmark::State& state) {
  runLocal(state);
}

BENCHMARK_DEFINE_F(PrefixSumBench, LocalInt)(benchmark::State& state) {
  runLocalInt(state);
}

BENCHMARK_DEFINE_F(PrefixSumBench, NNC)(benchmark::State& state) {
  runNNC(state);
}

#ifdef __AVX2__
BENCHMARK_DEFINE_F(PrefixSumBench, LocalAVX2)(benchmark::State& state) {
  runLocalAVX2(state);
}
BENCHMARK_DEFINE_F(PrefixSumBench, LocalIntAVX2)(benchmark::State& state) {
  runLocalIntAVX2(state);
}
#endif

#ifdef __AVX512F__
BENCHMARK_DEFINE_F(PrefixSumBench, LocalAVX512)(benchmark::State& state) {
  runLocalAVX512(state);
}
BENCHMARK_DEFINE_F(PrefixSumBench, LocalIntAVX512)(benchmark::State& state) {
  runLocalIntAVX512(state);
}

BENCHMARK_DEFINE_F(PrefixSumBench, ExclusiveScanAVX512)
(benchmark::State& state) {
  runExclusiveScanAVX512(state);
}
BENCHMARK_DEFINE_F(PrefixSumBench, ExclusiveScanIntAVX512)
(benchmark::State& state) {
  runExclusiveScanIntAVX512(state);
}
#endif

//---------- float benchmarks ----------//
BENCHMARK_REGISTER_F(PrefixSumBench, ATen)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});

BENCHMARK_REGISTER_F(PrefixSumBench, NNC)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});

BENCHMARK_REGISTER_F(PrefixSumBench, Local)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});

#ifdef __AVX2__
BENCHMARK_REGISTER_F(PrefixSumBench, LocalAVX2)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});
#endif

#ifdef __AVX512F__
BENCHMARK_REGISTER_F(PrefixSumBench, LocalAVX512)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});
BENCHMARK_REGISTER_F(PrefixSumBench, ExclusiveScanAVX512)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});
#endif

//---------- int benchmarks ----------//
BENCHMARK_REGISTER_F(PrefixSumBench, LocalInt)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});

#ifdef __AVX2__
BENCHMARK_REGISTER_F(PrefixSumBench, LocalIntAVX2)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});
#endif

#ifdef __AVX512F__
BENCHMARK_REGISTER_F(PrefixSumBench, LocalIntAVX512)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});
BENCHMARK_REGISTER_F(PrefixSumBench, ExclusiveScanIntAVX512)
    ->RangeMultiplier(4)
    ->Ranges({{1 << 6, 1 << 20}});
#endif
