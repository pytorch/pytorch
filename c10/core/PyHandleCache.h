#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <c10/macros/Macros.h>
#include <c10/util/python_stub.h>

#include <atomic>

namespace c10 {

// A PyHandleCache represents a cached pointer from a C++ object to
// a Python object that represents that object analogously in Python.
// Upon a cache hit, the relevant object can be retrieved after a test
// and then a memory load.  Two conditions must hold to be able to use this
// class:
//
//  - This must truly be a cache; e.g., the caller must be able to produce
//    the object some other way if the cache hit misses.
//
//  - This must truly be a handle; e.g., the Python object referenced by
//    this class must have static lifetime.  This means we don't have to
//    maintain strong ownership or deallocate the object when the C++ object
//    dies.  Static lifetime is a good idea in conjunction with the cache,
//    since if you are producing a fresh object on miss you won't be
//    maintaining object identity.  If you need bidirectional ownership,
//    you will want to factor out the pattern in TensorImpl with
//    resurrection.
//
// This cache is expected to not improve perf under torchdeploy, as one
// interpreter will fill up the cache, and all the interpreters will be
// unable to use the slot.  A potential improvement is to have multiple
// slots (one per interpreter), which will work in deployment scenarios
// where there a stable, fixed number of interpreters.  You can also store
// the relevant state in the Python library, rather than in the non-Python
// library (although in many cases, this is not convenient, as there may
// not be a way to conveniently index based on the object.)
class PyHandleCache {
 public:
  PyHandleCache() : pyinterpreter_(nullptr), data_(nullptr) {}

  // Attempt to fetch the pointer from the cache, if the PyInterpreter
  // matches.  If it doesn't exist, or the cache entry is not valid,
  // use slow_accessor to get the real pointer value and return that
  // (possibly writing it to the cache, if the cache entry is
  // available.)
  template <typename F>
  PyObject* ptr_or(impl::PyInterpreter* self_interpreter, F slow_accessor)
      const {
    // Note [Memory ordering on Python interpreter tag]
    impl::PyInterpreter* interpreter =
        pyinterpreter_.load(std::memory_order_acquire);
    if (C10_LIKELY(interpreter == self_interpreter)) {
      return data_;
    } else if (interpreter == nullptr) {
      auto* r = slow_accessor();
      impl::PyInterpreter* expected = nullptr;
      // attempt to claim this cache entry with the specified interpreter tag
      if (pyinterpreter_.compare_exchange_strong(
              expected, self_interpreter, std::memory_order_acq_rel)) {
        data_ = r;
      }
      // This shouldn't be possible, as you should be GIL protected
      TORCH_INTERNAL_ASSERT(expected != self_interpreter);
      return r;
    } else {
      return slow_accessor();
    }
  }

 private:
  mutable std::atomic<impl::PyInterpreter*> pyinterpreter_;
  mutable PyObject* data_;
};

} // namespace c10
