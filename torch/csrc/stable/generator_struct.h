#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/device_struct.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <memory>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

// The torch::stable::Generator class is a highlevel C++ wrapper around the C
// shim Generator APIs, modeled after at::Generator (see ATen/core/Generator.h).
//
// Like at::Generator, this is a value-semantics handle to a refcounted RNG
// state: copies share the same underlying generator. We back it with a
// shared_ptr<AtenGeneratorOpaque> (an owning AtenGeneratorHandle), mirroring
// torch::stable::Tensor. This faithfully reflects at::Generator's own shared
// (intrusive_ptr) semantics and avoids forcing move-only ownership on callers.

/**
 * @brief An ABI stable wrapper around a PyTorch Generator.
 *
 * Modeled after at::Generator. Copies share the same underlying RNG state.
 *
 * Minimum compatible version: PyTorch 2.13.
 */
class Generator {
 private:
  std::shared_ptr<AtenGeneratorOpaque> gen_;

 public:
  /**
   * @brief Constructs a Generator from an existing AtenGeneratorHandle.
   *
   * Steals ownership of the provided AtenGeneratorHandle.
   *
   * @param gen The AtenGeneratorHandle to wrap. Ownership is transferred to
   *            this Generator.
   *
   * Minimum compatible version: PyTorch 2.13.
   */
  explicit Generator(AtenGeneratorHandle gen)
      : gen_(gen, [](AtenGeneratorHandle gen) {
          STABLE_TORCH_ERROR_CODE_CHECK(torch_delete_generator(gen));
        }) {}

  // Copy and move can be default cuz the underlying handle is a shared_ptr
  /// \private
  Generator(const Generator& other) = default;
  /// \private
  Generator(Generator&& other) noexcept = default;
  /// \private
  Generator& operator=(const Generator& other) = default;
  /// \private
  Generator& operator=(Generator&& other) noexcept = default;
  /// \private
  ~Generator() = default;

  /**
   * @brief Returns a borrowed reference to the underlying AtenGeneratorHandle.
   *
   * @return The underlying AtenGeneratorHandle.
   *
   * Minimum compatible version: PyTorch 2.13.
   */
  AtenGeneratorHandle get() const {
    return gen_.get();
  }

  // defined in generator_inl.h to avoid circular dependencies
  /**
   * @brief Returns the device of the generator.
   *
   * @return The Device on which the generator resides.
   *
   * Minimum compatible version: PyTorch 2.13.
   */
  Device device() const;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

HIDDEN_NAMESPACE_END(torch, stable)
