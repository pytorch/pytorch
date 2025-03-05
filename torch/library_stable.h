// this file can only have stable stuff! Akin to shim.h
// but unlike shim.h, this file can contain header-only C++
// code for better UX.

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

class TORCH_API StableLibrary final {
  private:
    TorchLibraryHandle lib_;
  public:
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
        const char* ns,
        const char* k,
        const char* file,
        uint32_t line) {
      if (kind==IMPL) {
        aoti_torch_library_init_for_impl(ns, k, file, line, &lib_);
      } else {
        std::cout << "ERROR: StableLibrary constructor not yet implemented for kind=" << kind << std::endl;
      }
    }

    StableLibrary(const StableLibrary&) = delete;
    StableLibrary& operator=(const StableLibrary&) = delete;
    StableLibrary(StableLibrary&&) = default;
    StableLibrary& operator=(StableLibrary&&) = default;
    ~StableLibrary() = default;

    StableLibrary& impl(const char* name, void (*fn)(uint64_t*, int64_t, int64_t)) {
      aoti_torch_library_impl(lib_, name, fn);
      return *this;
    }
};

class TORCH_API StableTorchLibraryInit final {
  private:
    using InitFn = void(StableLibrary&);
    StableLibrary lib_;

  public:
    StableTorchLibraryInit(
      StableLibrary::Kind kind,
      InitFn* fn,
      const char* ns,
      const char* k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) {
        fn(lib_);
      }
};


// macros copied from c10/macros/Macros.h
#ifdef __COUNTER__
#define STABLE_UID __COUNTER__
#else
#define STABLE_UID __LINE__
#endif

#define STABLE_CONCATENATE_IMPL(s1, s2) s1##s2
#define STABLE_CONCATENATE(s1, s2) STABLE_CONCATENATE_IMPL(s1, s2)
// end of macros copied from c10/macros/Macros.h

#define STABLE_TORCH_LIBRARY_IMPL(ns, k, m) _STABLE_TORCH_LIBRARY_IMPL(ns, k, m, STABLE_UID)

#define _STABLE_TORCH_LIBRARY_IMPL(ns, k, m, uid)                         \
  static void STABLE_CONCATENATE(                                            \
      STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(StableLibrary&);       \
  static const StableTorchLibraryInit STABLE_CONCATENATE(           \
      STABLE_TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(                 \
      StableLibrary::IMPL,                                               \
      &STABLE_CONCATENATE(STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid), \
      #ns,                                                                \
      #k,                            \
      __FILE__,                                                           \
      __LINE__);                                                          \
  void STABLE_CONCATENATE(                                                   \
      STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(StableLibrary & m)
