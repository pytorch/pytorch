#pragma once

// The InputBuffer class accumulates a list of Variables for use by a
// function. It implements logic to avoid modifying the passed
// values in-place (adding an input twice will accumulate the result).
// This behaviour is needed and used only in backward graphs.

#include <utility>
#include <vector>

#include <c10/core/Stream.h>
#include <torch/csrc/autograd/variable.h>
#include <optional>

namespace torch::autograd {

// Detect and handle stale non-capturing stream references during CUDA graph
// capture. Autograd nodes snapshot the current stream at construction time.
// If a node was created on a non-capturing stream (e.g. during warmup) but
// backward runs under capture on a different stream, the engine would issue
// cudaStreamWaitEvent on the stale stream, pulling it into the capture.
//
// Three possible outcomes, depending on global state:
//   1. If overrideStaleCaptureStream() is true, returns capturing_stream
//      (the caller should use it in place of node_stream).
//   2. If node_stream refers to the default stream (id == 0), throws a
//      c10::Error with actionable guidance; this case always invalidates
//      the capture, so failing fast is better than the opaque CUDA error.
//   3. Otherwise, returns node_stream unchanged.
//
// Preconditions (asserted): node_stream is set and non-capturing;
// capturing_stream is set and capturing. Intended as an internal helper,
// not part of the stable C++ ABI.
std::optional<c10::Stream> maybe_override_stale_capture_stream(
    const std::optional<c10::Stream>& node_stream,
    const std::optional<c10::Stream>& capturing_stream,
    const std::string& node_name);

struct InputBuffer {
  explicit InputBuffer(size_t size)
      : buffer(size),
        opt_accum_streams(size),
        ready_events(size),
        ready_streams(size) {}
  InputBuffer(const InputBuffer& other) = delete;
  InputBuffer(InputBuffer&& other) = default;
  explicit InputBuffer(variable_list&& inputs) : buffer(std::move(inputs)) {}
  InputBuffer& operator=(InputBuffer&& other) = default;

  // Accumulates the variable at a specified index.
  // The optional CUDA streams determine which stream the accumulation
  // is run on and how the addition is synchronized.
  TORCH_API void add(
      size_t pos,
      Variable&& var,
      const std::optional<c10::Stream>& opt_producer_stream,
      const std::optional<c10::Stream>& opt_consumer_stream,
      Node* fn);

  Variable operator[](size_t pos) {
    return buffer[pos];
  }

  // Returns the inputs as a list of variables. Destroys given InputBuffer.
  static std::vector<Variable> variables(InputBuffer&& g);

  std::vector<Variable> buffer;
  // The stream used for accumulation when a variable is used multiple times.
  std::vector<std::optional<c10::Stream>> opt_accum_streams;
  // The events you need to wait for to ensure the corresponding buffers
  // are ready. The events are updated as we accumulate into the buffer.
  std::vector<std::optional<c10::Event>> ready_events;
  // The streams corresponding to the events above. This is only used to
  // check if more synchronization is needed or not.
  std::vector<std::optional<c10::Stream>> ready_streams;
  // The consumer stream after the stale-capture override fires. Set by
  // InputBuffer::add() the first time a producer triggers the override
  // (i.e. only when set_override_stale_capture_stream is true and a stale
  // non-capturing consumer is detected against a capturing producer).
  // Subsequent add() calls reuse this so all producers feeding this buffer
  // converge on the overridden stream, and Engine::evaluate_function uses
  // it as the parent stream for the node. When no override has fired this
  // is nullopt and Engine::evaluate_function falls back to the node's
  // func->stream().
  std::optional<c10::Stream> opt_overridden_consumer_stream;
};

} // namespace torch::autograd
