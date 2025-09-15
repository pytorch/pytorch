#pragma once

#include <chrono>
#include <random>
#include <thread>

#include <c10/macros/Macros.h>

namespace c10d {

class TORCH_API Backoff {
 public:
  virtual ~Backoff() = default;

  virtual std::chrono::milliseconds nextBackoff() = 0;
  virtual void reset() = 0;

  void sleepBackoff() {
    std::this_thread::sleep_for(nextBackoff());
  }
};

class TORCH_API ExponentialBackoffWithJitter : public Backoff {
 public:
  ExponentialBackoffWithJitter();

  std::chrono::milliseconds nextBackoff() override;
  void reset() override;

 public:
  std::chrono::milliseconds initialInterval{500};
  double randomizationFactor{0.5};
  double multiplier{1.5};
  std::chrono::milliseconds maxInterval{60000};

 private:
  std::mt19937 gen_;
  std::chrono::milliseconds currentInterval_{0};
};

class TORCH_API FixedBackoff : public Backoff {
 public:
  FixedBackoff(std::chrono::milliseconds interval);

  std::chrono::milliseconds nextBackoff() override;
  void reset() override;

 private:
  std::chrono::milliseconds interval_;
};

} // namespace c10d
