// this file can only have stable stuff! Akin to shim.h
// but unlike shim.h, this file can contain header-only C++
// code for better UX.

#include <torch/csrc/inductor/aoti_torch/c/shim.h>


// helpers for converting between StableIValue and actual IValues
using StableIValue = uint64_t;

template <typename T>
StableIValue from(T val) {
  static_assert(
      sizeof(T) <= sizeof(StableIValue),
      "StableLibrary stack does not support parameter types larger than 64 bits.");
  return *reinterpret_cast<StableIValue*>(&val);
}

template <typename T>
T to(StableIValue val) {
  return *reinterpret_cast<T*>(&val);
}
// end to helpers for converting between StableIValue and actual IValues


class TORCH_API StableLibrary final {
  private:
    TorchLibraryHandle lib_;
  public:
    enum Kind {
      DEF,
      IMPL,
      FRAGMENT,
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
      if (kind == IMPL) {
        aoti_init_torch_library_impl(ns, k, file, line, &lib_);
      } else if (kind == DEF) {
        aoti_init_torch_library_def(ns, file, line, &lib_);
      } else { // kind == FRAGMENT
        aoti_init_torch_library_fragment(ns, file, line, &lib_);
      }
    }

    StableLibrary(const StableLibrary&) = delete;
    StableLibrary& operator=(const StableLibrary&) = delete;
    StableLibrary(StableLibrary&&) = default;
    StableLibrary& operator=(StableLibrary&&) = default;
    ~StableLibrary() = default;

    StableLibrary& impl(const char* name, void (*fn)(StableIValue*, int64_t, int64_t)) {
      aoti_torch_library_impl(lib_, name, fn);
      return *this;
    }

    StableLibrary& def(const char* name) {
      aoti_torch_library_def(lib_, name);
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
      StableLibrary::Kind::IMPL,                                               \
      &STABLE_CONCATENATE(STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid), \
      #ns,                                                                \
      #k,                            \
      __FILE__,                                                           \
      __LINE__);                                                          \
  void STABLE_CONCATENATE(                                                   \
      STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(StableLibrary & m)


#define STABLE_TORCH_LIBRARY(ns, m)                                                   \
  static void STABLE_TORCH_LIBRARY_init_##ns(StableLibrary&);                        \
  static const StableTorchLibraryInit STABLE_TORCH_LIBRARY_static_init_##ns( \
      StableLibrary::Kind::DEF,                                                     \
      &STABLE_TORCH_LIBRARY_init_##ns,                                                \
      #ns,                                                                     \
      nullptr,                                                            \
      __FILE__,                                                                \
      __LINE__);                                                               \
  void STABLE_TORCH_LIBRARY_init_##ns(StableLibrary& m)


#define STABLE_TORCH_LIBRARY_FRAGMENT(ns, m) _STABLE_TORCH_LIBRARY_FRAGMENT(ns, m, STABLE_UID)

#define _STABLE_TORCH_LIBRARY_FRAGMENT(ns, m, uid)                       \
  static void STABLE_CONCATENATE(                                    \
      STABLE_TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid)(StableLibrary&); \
  static const StableTorchLibraryInit STABLE_CONCATENATE(   \
      STABLE_TORCH_LIBRARY_FRAGMENT_static_init_##ns##_, uid)(           \
      StableLibrary::Kind::FRAGMENT,                                   \
      &STABLE_CONCATENATE(STABLE_TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid), \
      #ns,                                                        \
      nullptr,                                               \
      __FILE__,                                                   \
      __LINE__);                                                  \
  void STABLE_CONCATENATE(                                           \
      STABLE_TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid)(StableLibrary & m)
