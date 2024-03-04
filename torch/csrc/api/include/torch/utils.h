#pragma once

#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/profiler.h>
#include <cstdint>

namespace torch {

/// A RAII, thread-local guard that disabled gradient calculation.
///
/// Disabling gradient calculation is useful for inference, when you are sure
/// that you will not call `at::Tensor::backward`. It will reduce memory
/// consumption for computations that would otherwise have `requires_grad() ==
/// true`.
///
/// In this mode, the result of every computation will have
/// `requires_grad() == false`, even when the inputs have `requires_grad() ==
/// true`.
///
/// This context manager is thread-local; it will not affect computation
/// in other threads.
///
/// Example:
/// @code
/// auto x = torch::tensor({1.}, torch::requires_grad());
/// {
///   torch::NoGradGuard no_grad;
///   auto y = x * 2;
///   std::cout << y.requires_grad() << std::endl; // prints `false`
/// }
/// {
///   auto doubler = [](torch::Tensor x) {
///     torch::NoGradGuard no_grad;
///     return x * 2;
///   };
///   auto z = doubler(x);
///   std::cout << z.requires_grad() << std::endl; // prints `false`
/// }
/// @endcode
using NoGradGuard = at::NoGradGuard;

/// A RAII, thread-local guard that sets gradient calculation to on or off.
///
/// ``AutoGradMode`` will enable or disable grads based on its argument
/// `enabled`.
///
/// This context manager is thread-local; it will not affect computation
/// in other threads.
///
/// \param enabled: Flag whether to enable grad (``true``), or disable
///              (``false``). This can be used to conditionally enable
///              gradients.
///
/// Example:
/// @code
/// auto x = torch::tensor({1.}, torch::requires_grad());
/// {
///   torch::AutoGradMode enable_grad(true);
///   auto y = x * 2;
///   std::cout << y.requires_grad() << std::endl; // prints `true`
/// }
/// {
///   torch::AutoGradMode enable_grad(false);
///   auto y = x * 2;
///   std::cout << y.requires_grad() << std::endl; // prints `false`
/// }
/// @endcode
using AutoGradMode = at::AutoGradMode;

/// Sets the global random seed for all newly created CPU and CUDA tensors.
using at::manual_seed;

// Called during new thread initialization
using at::init_num_threads;

// Returns the number of threads used in parallel region.
using at::get_num_threads;

// Sets the number of threads to be used in parallel region.
using at::set_num_threads;

// Returns the number of threads used for inter-op parallelism.
using at::get_num_interop_threads;

// Sets the number of threads to be used for inter-op parallelism.
using at::set_num_interop_threads;

// Returns true if both t1, t2 are undefined or both are defined and equal
inline bool equal_if_defined(Tensor t1, Tensor t2) {
  return (
      (!t1.defined() && !t2.defined()) ||
      (t1.defined() && t2.defined() && torch::equal(t1, t2)));
}

// RecordFunction API
using at::addGlobalCallback;
using at::addThreadLocalCallback;
using at::CallbackHandle;
using at::clearCallbacks;
using at::clearGlobalCallbacks;
using at::clearThreadLocalCallbacks;
using at::DisableRecordFunctionGuard;
using at::enableRecordFunction;
using at::hasCallbacks;
using at::hasGlobalCallbacks;
using at::hasThreadLocalCallbacks;
using at::isRecordFunctionEnabled;
using at::RecordFunction;
using at::RecordFunctionCallback;
using at::RecordFunctionGuard;
using at::removeCallback;

} // namespace torch
