// this file can only have stable stuff! Akin to shim.h

#include <c10/macros/Macros.h>     // used for C10_UID, verified to be header-only
#include <c10/core/DispatchKey.h>  // used for DispatchKey, enum verified to be header-only
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <optional>
#include <string>

class StableLibrary final {
  private:
    class TorchLibraryOpaque;
    using TorchLibraryHandle = TorchLibraryOpaque*;
    TorchLibraryHandle lib_;  // pimpl unique_ptr
  public:
    // a pointer to a real Library
    // a kind
    enum Kind {
      // DEF, // from TORCH_LIBRARY (no qualifier)
      IMPL,
      // FRAGMENT,
    };

    // constructor
    /// \private
    ///
    /// Use STABLE_TORCH_LIBRARY or STABLE_TORCH_LIBRARY_IMPL() instead of using these
    /// constructors directly
    StableLibrary(
        Kind kind,
        std::string ns,
        std::optional<c10::DispatchKey> k,
        const char* file,
        uint32_t line);

    StableLibrary(const StableLibrary&) = delete;
    StableLibrary& operator=(const StableLibrary&) = delete;
    StableLibrary(StableLibrary&&) = default;
    StableLibrary& operator=(StableLibrary&&) = default;
    ~StableLibrary() = default;

    StableLibrary& impl(const char* name, void (*fn)(void **, int64_t, int64_t));
};

class StableTorchLibraryInit final {
  private:
    using InitFn = void(StableLibrary&);
    StableLibrary lib_;

  public:
    StableTorchLibraryInit(
      StableLibrary::Kind kind,
      InitFn* fn,
      const char* ns,
      std::optional<c10::DispatchKey> k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) {
        fn(lib_);
      }
};


#define STABLE_TORCH_LIBRARY_IMPL(ns, k, m) _STABLE_TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)

#define _STABLE_TORCH_LIBRARY_IMPL(ns, k, m, uid)                         \
  static void C10_CONCATENATE(                                            \
      STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(StableLibrary&);       \
  static const StableTorchLibraryInit C10_CONCATENATE(           \
      STABLE_TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(                 \
      StableLibrary::IMPL,                                               \
      &C10_CONCATENATE(STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid), \
      #ns,                                                                \
      std::make_optional(c10::DispatchKey::k),                            \
      __FILE__,                                                           \
      __LINE__);                                                          \
  void C10_CONCATENATE(                                                   \
      STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(StableLibrary & m)
