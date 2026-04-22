#include "profiler/OpenRegTracer.h"

#include <stack>
#include <utility>

namespace torch_openreg::profiler {

namespace {

// Thread-local correlation ID stack.  PyTorch pushes an ID before each
// ATen operator dispatch and pops it after, so device events recorded
// in between inherit the correct correlation.
struct CorrelationState {
  std::stack<uint64_t> stack;
};

CorrelationState& tls_correlation() {
  thread_local CorrelationState state;
  return state;
}

} // namespace

// --- singleton ---

OpenRegTracer& OpenRegTracer::instance() {
  static OpenRegTracer tracer;
  return tracer;
}

// --- enable / disable ---

void OpenRegTracer::enable() {
  enabled_.store(true, std::memory_order_release);
}

void OpenRegTracer::disable() {
  enabled_.store(false, std::memory_order_release);
}

bool OpenRegTracer::isEnabled() const {
  return enabled_.load(std::memory_order_acquire);
}

// --- correlation stack ---

void OpenRegTracer::pushCorrelation(uint64_t id) {
  tls_correlation().stack.push(id);
}

void OpenRegTracer::popCorrelation() {
  auto& s = tls_correlation().stack;
  if (!s.empty()) {
    s.pop();
  }
}

uint64_t OpenRegTracer::currentCorrelation() const {
  auto& s = tls_correlation().stack;
  return s.empty() ? 0 : s.top();
}

// --- record collection ---

void OpenRegTracer::record(TraceRecord rec) {
  std::lock_guard<std::mutex> lock(mu_);
  records_.push_back(std::move(rec));
}

std::vector<TraceRecord> OpenRegTracer::flush() {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<TraceRecord> out;
  out.swap(records_);
  return out;
}

} // namespace torch_openreg::profiler
