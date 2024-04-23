// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/span_id.h"
#include "opentelemetry/trace/trace_flags.h"
#include "opentelemetry/trace/trace_id.h"
#include "opentelemetry/trace/trace_state.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{
/* SpanContext contains the state that must propagate to child Spans and across
 * process boundaries. It contains TraceId and SpanId, TraceFlags, TraceState
 * and whether it was propagated from a remote parent.
 */
class SpanContext final
{
public:
  /* A temporary constructor for an invalid SpanContext.
   * Trace id and span id are set to invalid (all zeros).
   *
   * @param sampled_flag a required parameter specifying if child spans should be
   * sampled
   * @param is_remote true if this context was propagated from a remote parent.
   */
  SpanContext(bool sampled_flag, bool is_remote) noexcept
      : trace_id_(),
        span_id_(),
        trace_flags_(trace::TraceFlags((uint8_t)sampled_flag)),
        is_remote_(is_remote),
        trace_state_(TraceState::GetDefault())
  {}

  SpanContext(TraceId trace_id,
              SpanId span_id,
              TraceFlags trace_flags,
              bool is_remote,
              nostd::shared_ptr<TraceState> trace_state = TraceState::GetDefault()) noexcept
      : trace_id_(trace_id),
        span_id_(span_id),
        trace_flags_(trace_flags),
        is_remote_(is_remote),
        trace_state_(trace_state)
  {}

  SpanContext(const SpanContext &ctx) = default;

  // @returns whether this context is valid
  bool IsValid() const noexcept { return trace_id_.IsValid() && span_id_.IsValid(); }

  // @returns the trace_flags associated with this span_context
  const trace::TraceFlags &trace_flags() const noexcept { return trace_flags_; }

  // @returns the trace_id associated with this span_context
  const trace::TraceId &trace_id() const noexcept { return trace_id_; }

  // @returns the span_id associated with this span_context
  const trace::SpanId &span_id() const noexcept { return span_id_; }

  // @returns the trace_state associated with this span_context
  const nostd::shared_ptr<trace::TraceState> trace_state() const noexcept { return trace_state_; }

  /*
   * @param that SpanContext for comparing.
   * @return true if `that` equals the current SpanContext.
   * N.B. trace_state is ignored for the comparison.
   */
  bool operator==(const SpanContext &that) const noexcept
  {
    return trace_id() == that.trace_id() && span_id() == that.span_id() &&
           trace_flags() == that.trace_flags();
  }

  SpanContext &operator=(const SpanContext &ctx) = default;

  bool IsRemote() const noexcept { return is_remote_; }

  static SpanContext GetInvalid() noexcept { return SpanContext(false, false); }

  bool IsSampled() const noexcept { return trace_flags_.IsSampled(); }

private:
  trace::TraceId trace_id_;
  trace::SpanId span_id_;
  trace::TraceFlags trace_flags_;
  bool is_remote_;
  nostd::shared_ptr<trace::TraceState> trace_state_;
};
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
