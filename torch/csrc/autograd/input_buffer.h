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
      const std::optional<c10::Stream>& opt_consumer_stream);

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
};

} // namespace torch::autograd
