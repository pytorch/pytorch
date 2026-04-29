#pragma once

namespace c10 {

/**
 * Note [Flags defining the behavior of events]
 *
 * PYTORCH_DEFAULT and BACKEND_DEFAULT are valid for all backends. The
 * BACKEND_DEFAULT is what a particular backend would select if no
 * flags were given. PYTORCH_DEFAULT is the PyTorch's framework default
 * choice for events on that backend, which may not be the same.
 *
 * The mapping of PYTORCH_DEFAULT and BACKEND_DEFAULT is done by each
 * backend implementation.
 */
enum class EventFlag {
  // Disable timing
  PYTORCH_DEFAULT,
  // Enable timing
  BACKEND_DEFAULT,
  // FOR TESTING ONLY
  INVALID
};

} // namespace c10
