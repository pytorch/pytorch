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

extern "C" {

orError_t orActivityEnableTracing() {
  openreg::profiler::OpenRegTracer::instance().enableActivityTracing();
  return orSuccess;
}

orError_t orActivityDisableTracing() {
  openreg::profiler::OpenRegTracer::instance().disableActivityTracing();
  return orSuccess;
}

orError_t orActivityPushExternalCorrelationId(uint64_t id) {
  openreg::profiler::OpenRegTracer::instance().pushExternalCorrelationId(id);
  return orSuccess;
}

orError_t orActivityPopExternalCorrelationId(uint64_t* id) {
  if (id) {
    *id = openreg::profiler::OpenRegTracer::instance()
              .getExternalCorrelationId();
  }
  openreg::profiler::OpenRegTracer::instance().popExternalCorrelationId();
  return orSuccess;
}

} // extern "C"