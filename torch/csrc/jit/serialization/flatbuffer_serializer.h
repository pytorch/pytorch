#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/jit/mobile/module.h>

/**
 * Defines the public API for serializing mobile modules to flatbuffer.
 * Note that this header must not include or depend on flatbuffer-defined
 * types, to avoid leaking those details to PyTorch clients.
 */

namespace torch {
namespace jit {

/// Maps file names to file contents.
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

/**
 * Represents a span of data. Typically owned by a UniqueDetachedBuffer.
 */
class TORCH_API DetachedBuffer final {
 public:
  /// Creates a new DetachedBuffer with an optional data owner. This interface
  /// is provided to let users create objects of this type for testing.
  DetachedBuffer(void* data, size_t size, void* internal_data_owner = nullptr)
      : data_(data), size_(size), data_owner_(internal_data_owner) {}

  /// Returns a pointer to the data.
  C10_NODISCARD void* data() {
    return data_;
  }
  /// Returns a pointer to the data.
  C10_NODISCARD const void* data() const {
    return data_;
  }
  /// Returns the size of the data, in bytes.
  C10_NODISCARD size_t size() const {
    return size_;
  }

  /// Wrapper type that typically owns data_owner_.
  using UniqueDetachedBuffer =
      std::unique_ptr<DetachedBuffer, std::function<void(DetachedBuffer*)>>;

 private:
  /// Deletes the owner, if present, and the buf itself.
  /// Note: we could have provided a movable type with a destructor that did
  /// this work, but the unique wrapper was easier in practice.
  static void destroy(DetachedBuffer* buf);

  /// Provides access to destroy() for implementation and testing.
  friend struct DetachedBufferFriend;
  friend struct DetachedBufferTestingFriend;

  /// Pointer to the data. Not owned by this class.
  void* data_;
  /// The size of `data_`, in bytes.
  size_t size_;
  /// Opaque pointer to the underlying owner of `data_`. This class
  /// (DetachedBuffer) does not own the owner or the data. It will typically be
  /// owned by a UniqueDetachedBuffer that knows how to delete the owner along
  /// with this class.
  void* data_owner_;
};

TORCH_API void save_mobile_module(
    const mobile::Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files = ExtraFilesMap(),
    const ExtraFilesMap& jit_sources = ExtraFilesMap(),
    const std::vector<IValue>& jit_constants = {});

TORCH_API DetachedBuffer::UniqueDetachedBuffer save_mobile_module_to_bytes(
    const mobile::Module& module,
    const ExtraFilesMap& extra_files = ExtraFilesMap(),
    const ExtraFilesMap& jit_sources = ExtraFilesMap(),
    const std::vector<IValue>& jit_constants = {});

// This function will make the capabilities to load and safe
// Module as a flatbuffer file available for use by _load_for_mobile
// and friends. This is NOT needed if using the other functions
// in this file directly.
TORCH_API bool register_flatbuffer_serializer();

} // namespace jit
} // namespace torch
