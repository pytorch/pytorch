#include <c10/util/intrusive_ptr.h>

#include "benchmark/benchmark.h"

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>

using c10::intrusive_ptr;
using c10::intrusive_ptr_target;
using c10::make_intrusive;
using c10::weak_intrusive_ptr;

namespace {

// Foo uses intrusive ptr
class Foo : public intrusive_ptr_target {
 public:
  Foo(int param_) : param(param_) {}
  int param;
};


class Bar : public std::enable_shared_from_this<Bar> {
 public:
  Bar(int param_) : param(param_) {}
  int param;
};

static void BM_IntrusivePtrCtorDtor(benchmark::State& state) {
  intrusive_ptr<Foo> var = make_intrusive<Foo>(0);
  while (state.KeepRunning()) {
    volatile intrusive_ptr<Foo> var2 = var;
  }
}
BENCHMARK(BM_IntrusivePtrCtorDtor);

static void BM_SharedPtrCtorDtor(benchmark::State& state) {
  std::shared_ptr<Bar> var = std::make_shared<Bar>(0);
  while (state.KeepRunning()) {
    volatile std::shared_ptr<Bar> var2 = var;
  }
}
BENCHMARK(BM_SharedPtrCtorDtor);

// todo: parameterize the array length
static const int kLength = 1000;

static void BM_IntrusivePtrArray(benchmark::State& state) {
  intrusive_ptr<Foo> var = make_intrusive<Foo>(0);
  std::vector<intrusive_ptr<Foo> > vararray(kLength);
  while (state.KeepRunning()) {
    for (int i = 0; i < kLength; ++i) {
      vararray[i] = var;
    }
    for (int i = 0; i < kLength; ++i) {
      vararray[i].reset();
    }
  }
}
BENCHMARK(BM_IntrusivePtrArray);

static void BM_SharedPtrArray(benchmark::State& state) {
  std::shared_ptr<Bar> var = std::make_shared<Bar>(0);
  std::vector<std::shared_ptr<Bar> > vararray(kLength);
  while (state.KeepRunning()) {
    for (int i = 0; i < kLength; ++i) {
      vararray[i] = var;
    }
    for (int i = 0; i < kLength; ++i) {
      vararray[i].reset();
    }
  }
}
BENCHMARK(BM_SharedPtrArray);
} // namespace


BENCHMARK_MAIN();
