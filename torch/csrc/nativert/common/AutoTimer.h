/*
 * Ported from folly/logging/AutoTimer.h
 */

#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <string_view>

#include <c10/util/Logging.h>
#include <fmt/format.h>

namespace torch::nativert {

// Default logger
enum class GoogleLoggerStyle { SECONDS, MILLISECONDS };
template <GoogleLoggerStyle>
struct GoogleLogger;

/**
 * Automatically times a block of code, printing a specified log message on
 * destruction or whenever the log() method is called. For example:
 *
 *   AutoTimer t("Foo() completed");
 *   doWork();
 *   t.log("Do work finished");
 *   doMoreWork();
 *
 * This would print something like:
 *   "Do work finished in 1.2 seconds"
 *   "Foo() completed in 4.3 seconds"
 *
 * You can customize what you use as the logger and clock. The logger needs
 * to have an operator()(StringPiece, std::chrono::duration<double>) that
 * gets a message and a duration. The clock needs to model Clock from
 * std::chrono.
 *
 * The default logger logs usings glog. It only logs if the message is
 * non-empty, so you can also just use this class for timing, e.g.:
 *
 *   AutoTimer t;
 *   doWork()
 *   const auto how_long = t.log();
 */
template <
    class Logger = GoogleLogger<GoogleLoggerStyle::MILLISECONDS>,
    class Clock = std::chrono::high_resolution_clock>
class AutoTimer final {
 public:
  using DoubleSeconds = std::chrono::duration<double>;

  explicit AutoTimer(
      std::string&& msg = "",
      const DoubleSeconds& minTimetoLog = DoubleSeconds::zero(),
      Logger&& logger = Logger())
      : destructionMessage_(std::move(msg)),
        minTimeToLog_(minTimetoLog),
        logger_(std::move(logger)) {}

  // It doesn't really make sense to copy AutoTimer
  // Movable to make sure the helper method for creating an AutoTimer works.
  AutoTimer(const AutoTimer&) = delete;
  AutoTimer(AutoTimer&&) = default;
  AutoTimer& operator=(const AutoTimer&) = delete;
  AutoTimer& operator=(AutoTimer&&) = default;

  ~AutoTimer() {
    if (destructionMessage_) {
      log(destructionMessage_.value());
    }
  }

  DoubleSeconds log(std::string_view msg = "") {
    return logImpl(Clock::now(), msg);
  }

  template <typename... Args>
  DoubleSeconds logFormat(fmt::format_string<Args...> fmt, Args&&... args) {
    auto now = Clock::now();
    return logImpl(now, fmt::format(fmt, std::forward<Args>(args)...));
  }

 private:
  // We take in the current time so that we don't measure time to call
  // to<std::string> or format() in the duration.
  DoubleSeconds logImpl(
      std::chrono::time_point<Clock> now,
      std::string_view msg) {
    auto duration = now - start_;
    if (duration >= minTimeToLog_) {
      logger_(msg, duration);
    }
    start_ = Clock::now(); // Don't measure logging time
    return duration;
  }

  std::optional<std::string> destructionMessage_;
  std::chrono::time_point<Clock> start_ = Clock::now();
  DoubleSeconds minTimeToLog_;
  Logger logger_;
};

template <GoogleLoggerStyle Style>
struct GoogleLogger final {
  void operator()(
      std::string_view msg,
      const std::chrono::duration<double>& sec) const {
    if (msg.empty()) {
      return;
    }
    if (Style == GoogleLoggerStyle::SECONDS) {
      LOG(INFO) << msg << " in " << sec.count() << " seconds";
    } else if (Style == GoogleLoggerStyle::MILLISECONDS) {
      LOG(INFO) << msg << " in "
                << std::chrono::duration_cast<
                       std::chrono::duration<double, std::milli>>(sec)
                       .count()
                << " milliseconds";
    }
  }
};

} // namespace torch::nativert
