//== nomnigraph/Support/Common.h - Common class implementations --*- C++ -*-==//
//
// TODO Licensing.
//
//===----------------------------------------------------------------------===//
//
// This file defines basic classes that are useful to inherit from.
//
//===----------------------------------------------------------------------===//

#ifndef NOM_SUPPORT_COMMON_H
#define NOM_SUPPORT_COMMON_H

#include <functional>
#include <list>

// These #defines are useful when writing passes as the collapse
//
// if (!cond) {
//   continue; // or break; or return;
// }
//
// into a single line without negation

#define NOM_REQUIRE_OR_(_cond, _expr) \
  if (!(_cond)) {                     \
    _expr;                            \
  }

#define NOM_REQUIRE_OR_CONT(_cond) NOM_REQUIRE_OR_(_cond, continue)
#define NOM_REQUIRE_OR_BREAK(_cond) NOM_REQUIRE_OR_(_cond, break)
#define NOM_REQUIRE_OR_RET_NULL(_cond) NOM_REQUIRE_OR_(_cond, return nullptr)
#define NOM_REQUIRE_OR_RET_FALSE(_cond) NOM_REQUIRE_OR_(_cond, return false)
#define NOM_REQUIRE_OR_RET_TRUE(_cond) NOM_REQUIRE_OR_(_cond, return true)
#define NOM_REQUIRE_OR_RET(_cond) NOM_REQUIRE_OR_(_cond, return )

// Implements accessors for a generic type T. If the type is not
// specified (i.e., void template type) then the partial specification
// gives an empty type.
template <typename T = void>
class StorageType {
 public:
  StorageType(T&& data) : Data(std::move(data)) {}
  StorageType(const T& data) = delete;
  StorageType() {}

  const T& data() const {
    return Data;
  }
  T* mutableData() {
    return &Data;
  }
  void resetData(T&& data) {
    Data = std::move(data);
  }

 private:
  T Data;
};

template <>
class StorageType<> {};

/// \brief This class enables a listener pattern.
/// It is to be used with a "curious recursive pattern"
/// i.e. Derived : public Notifier<Derived> {}
template <typename T>
class Notifier {
 public:
  using Callback = std::function<void(T*)>;
  Notifier() {}

  Callback* registerDestructorCallback(Callback fn) {
    dtorCallbacks_.emplace_back(fn);
    return &dtorCallbacks_.back();
  }

  Callback* registerNotificationCallback(Callback fn) {
    notifCallbacks_.emplace_back(fn);
    return &notifCallbacks_.back();
  }

  void deleteCallback(std::list<Callback>& callbackList, Callback* toDelete) {
    for (auto i = callbackList.begin(); i != callbackList.end(); ++i) {
      if (&*i == toDelete) {
        callbackList.erase(i);
        break;
      }
    }
  }

  void deleteDestructorCallback(Callback* c) {
    deleteCallback(dtorCallbacks_, c);
  }

  void deleteNotificationCallback(Callback* c) {
    deleteCallback(notifCallbacks_, c);
  }

  /// \brief Notifies all listeners (`registerNotificationCallback`
  /// users) of an update.  Assumes the information of the update
  /// is encoded in the state of the derived class, thus only passing
  /// a pointer of type T* to the callback.
  void notify() {
    for (auto callback : notifCallbacks_) {
      callback(reinterpret_cast<T*>(this));
    }
  }

  virtual ~Notifier() {
    for (auto callback : dtorCallbacks_) {
      callback(reinterpret_cast<T*>(this));
    }
  }

 private:
  std::list<Callback> dtorCallbacks_;
  std::list<Callback> notifCallbacks_;
};

#endif /* NOM_SUPPORT_COMMON_H */
