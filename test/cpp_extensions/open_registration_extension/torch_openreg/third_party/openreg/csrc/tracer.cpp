#include "tracer.h"

#include <stack>

namespace openreg::profiler {

namespace {

struct CorrelationState {
  std::stack<uint64_t> stack;
};

CorrelationState& tls_correlation() {
  thread_local CorrelationState state;
  return state;
}

} // namespace

OpenRegTracer& OpenRegTracer::instance() {
  static OpenRegTracer tracer;
  return tracer;
}

void OpenRegTracer::enableActivityTracing() {
  enabled_.store(true, std::memory_order_release);
}

void OpenRegTracer::disableActivityTracing() {
  enabled_.store(false, std::memory_order_release);
}

bool OpenRegTracer::isActivityTracingEnabled() const {
  return enabled_.load(std::memory_order_acquire);
}

void OpenRegTracer::pushExternalCorrelationId(uint64_t id) {
  tls_correlation().stack.push(id);
}

void OpenRegTracer::popExternalCorrelationId() {
  auto& s = tls_correlation().stack;
  if (!s.empty()) {
    s.pop();
  }
}

uint64_t OpenRegTracer::getExternalCorrelationId() const {
  auto& s = tls_correlation().stack;
  return s.empty() ? 0 : s.top();
}

} // namespace openreg::profiler
