#ifndef THP_STORAGE_INC
#define THP_STORAGE_INC

#include <torch/csrc/Types.h>

#define THPStorageStr "torch.UntypedStorage"

namespace c10 {

template <>
struct MaybeOwnedTraits<c10::Storage> {
  using owned_type = c10::Storage;
  using borrow_type = c10::Storage;

  static borrow_type createBorrow(const owned_type& from) {
    return borrow_type(from);
  }

  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
    lhs.unsafeReleaseStorageImpl();
    lhs = borrow_type(rhs);
  }

  static void destroyBorrow(borrow_type& toDestroy) {
    toDestroy.unsafeReleaseStorageImpl(); // "leak" it, but it was already +0.
  }

  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;
  }

  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) {
    return true;
  }
};

template <>
struct ExclusivelyOwnedTraits<c10::Storage> {
  using repr_type = c10::Storage;
  using pointer_type = c10::Storage*;
  using const_pointer_type = const c10::Storage*;

  static repr_type nullRepr() {
    return c10::Storage();
  }

  template <class... Args>
  static repr_type createInPlace(Args&&... args) {
    return c10::Storage(std::forward<Args>(args)...);
  }

  static repr_type moveToRepr(c10::Storage&& x) {
    return std::move(x);
  }

  static c10::Storage take(c10::Storage& x) {
    return std::move(x);
  }

  static pointer_type getImpl(repr_type& x) {
    return &x;
  }

  static const_pointer_type getImpl(const repr_type& x) {
    return &x;
  }
};

} // namespace c10

struct THPStorage {
  PyObject_HEAD;
  c10::MaybeOwned<c10::Storage> cdata;
};

TORCH_PYTHON_API PyObject* THPStorage_New(c10::Storage storage);
extern PyObject* THPStorageClass;

bool THPStorage_init(PyObject* module);
void THPStorage_postInit(PyObject* module);

extern PyTypeObject THPStorageType;

inline const c10::Storage& THPStorage_Unpack(THPStorage* storage) {
  return *storage->cdata;
}

inline const c10::Storage& THPStorage_Unpack(PyObject* obj) {
  return THPStorage_Unpack(reinterpret_cast<THPStorage*>(obj));
}

#endif
