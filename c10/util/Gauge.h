#pragma once

#include <memory>
#include <string_view>

#include <c10/macros/Macros.h>
#include <c10/util/SmallVector.h>

namespace c10::monitor {
namespace detail {

class GaugeImpl;

class GaugeBackendIf {
 public:
  virtual ~GaugeBackendIf() = default;
  virtual void record(int64_t value) noexcept = 0;
};

class GaugeBackendFactoryIf {
 public:
  virtual ~GaugeBackendFactoryIf() = default;

  // May return nullptr if the gauge will be ignored by the given backend.
  virtual std::unique_ptr<GaugeBackendIf> create(
      std::string_view key) noexcept = 0;
};

void C10_API registerGaugeBackend(std::unique_ptr<GaugeBackendFactoryIf>);
} // namespace detail

// A handle to a Gauge.
class C10_API GaugeHandle {
 public:
  explicit GaugeHandle(std::string_view key);
  void record(int64_t value);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  detail::GaugeImpl& impl_;
};

} // namespace c10::monitor

#define STATIC_GAUGE(_key)                            \
  []() -> ::c10::monitor::GaugeHandle& {              \
    static ::c10::monitor::GaugeHandle handle(#_key); \
    return handle;                                    \
  }()
