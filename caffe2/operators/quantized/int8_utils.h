#ifndef CAFFE2_INT8_UTILS_H_
#define CAFFE2_INT8_UTILS_H_

#include <gemmlowp/public/gemmlowp.h>

#include "caffe2/utils/threadpool/ThreadPool.h"
#include "caffe2/utils/threadpool/WorkersPool.h"

namespace caffe2 {

/*
 * Initialized QNNPACK (only once).
 * Throws if initialization failed.
 */
void initQNNPACK();

namespace int8 {

/*
 * Code here is partially derived from gemmlowp library
 * (https://github.com/google/gemmlowp)
 */

// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

inline int32_t MultiplyByQuantizedMultiplierSmallerThanOne(
    int32_t x,
    int32_t quantized_multiplier,
    int right_shift) {
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x, quantized_multiplier), right_shift);
}

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float Round(const float x) {
  return ::nearbyintf(x);
}
inline double Round(const double x) {
  return ::nearbyint(x);
}
#else
template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}
#endif

inline uint8_t QuantizeUint8(float scale, int32_t zero_point, float value) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();

  auto r = zero_point + static_cast<int32_t>(Round(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<uint8_t>(r);
}

inline void QuantizeMultiplierSmallerThanOne(
    double double_multiplier,
    int32_t* quantized_multiplier,
    int* right_shift) {
  CHECK(double_multiplier >= 0.);
  CHECK(double_multiplier < 1.);
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *right_shift = 0;
    return;
  }
  CHECK(double_multiplier > 0.);
  const double q = std::frexp(double_multiplier, right_shift);
  *right_shift *= -1;

  auto q_fixed = static_cast<int64_t>(Round(q * (1ll << 31)));
  CHECK(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    --*right_shift;
  }
  CHECK_GE(*right_shift, 0);
  CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

inline void QuantizeMultiplierGreaterThanOne(
    double double_multiplier,
    int32_t* quantized_multiplier,
    int* left_shift) {
  CHECK(double_multiplier > 1.);
  const double q = std::frexp(double_multiplier, left_shift);
  auto q_fixed = static_cast<int64_t>(Round(q * (1ll << 31)));
  CHECK(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*left_shift;
  }
  CHECK_GE(*left_shift, 0);
  CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

inline int32_t MultiplyByQuantizedMultiplierGreaterThanOne(
    int32_t x,
    int32_t quantized_multiplier,
    int left_shift) {
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  return SaturatingRoundingDoublingHighMul(
      x * (1 << left_shift), quantized_multiplier);
}

inline int CalculateInputRadius(int input_integer_bits, int input_left_shift) {
  const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
      (1ll << (31 - input_integer_bits)) / (1ll << input_left_shift);
  // Tighten bound using floor.  Suppose that we could use the exact value.
  // After scaling the difference, the result would be at the maximum.  Thus we
  // must ensure that our value has lower magnitude.
  return static_cast<int>(std::floor(max_input_rescaled));
}

// An adaptor to use the Caffe2 WorkersPool implementation for gemmlowp
// multithreading functions.
class C2GEMMContext : public gemmlowp::SingleThreadGemmContext {
  class C2WorkersPool;

 public:
  C2GEMMContext(ThreadPool* pool) : threadPool_(pool), workersPool_(pool) {}
  int max_num_threads() const {
    CHECK(threadPool_);
    return threadPool_->getNumThreads();
  }
  C2WorkersPool* workers_pool() {
    return &workersPool_;
  }

  ThreadPool* threadPool() {
    return threadPool_;
  }

 private:
  class C2WorkersPool {
   public:
    C2WorkersPool(ThreadPool* pool) : pool_(pool) {}
    void Execute(const std::vector<gemmlowp::Task*>& tasks) {
      class C2Task : public Task {
       public:
        C2Task(gemmlowp::Task* task) : task_(task){};
        virtual void Run() override {
          CHECK(task_);
          task_->Run();
        }

       private:
        gemmlowp::Task* task_;
      };
      std::vector<std::shared_ptr<Task>> c2tasks;
      c2tasks.reserve(tasks.size());
      std::vector<gemmlowp::Allocator> allocators(tasks.size());

      for (size_t i = 0; i < tasks.size(); ++i) {
        auto* task = tasks[i];
        CHECK(task);
        task->local_allocator = &allocators[i];
        c2tasks.push_back(std::shared_ptr<Task>(new C2Task(task)));
      }
      CHECK(pool_);
      pool_->withPool([&](WorkersPool* pool) { pool->Execute(c2tasks); });
      for (auto* t : tasks) {
        delete t;
      }
    }

   private:
    ThreadPool* pool_;
  };
  ThreadPool* threadPool_;
  C2WorkersPool workersPool_;
};

enum class Activation : uint8_t { NONE = 0, RELU = 1 };

inline std::pair<uint8_t, uint8_t>
activationLimits(float scale, int32_t zero_point, Activation Ac) {
  switch (Ac) {
    case Activation::NONE:
      return {std::numeric_limits<uint8_t>::min(),
              std::numeric_limits<uint8_t>::max()};
    case Activation::RELU:
      return {QuantizeUint8(scale, zero_point, 0.0),
              std::numeric_limits<uint8_t>::max()};
    default:
      __builtin_unreachable();
  }
}

} // namespace int8
} // namespace caffe2

#endif // CAFFE2_INT8_UTILS_H_
