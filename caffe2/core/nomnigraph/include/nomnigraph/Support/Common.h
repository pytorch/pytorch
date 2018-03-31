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

template <typename T> class StorageType {
public:
  StorageType(T &&data) : Data(std::move(data)) {}
  StorageType(const T &data) = delete;
  StorageType() {}

  const T &data() const { return Data; }
  T *mutableData() { return &Data; }
  void resetData(T &&data) { Data = std::move(data); }

private:
  T Data;
};

/// \brief This class enables a listener pattern.
/// It is to be used with a "curious recursive pattern"
/// i.e. Derived : public Notifier<Derived> {}
template <typename T> class Notifier {
public:
  using Callback = std::function<void(T*)>;
  Notifier() {}

  Callback* registerDestructorCallback(Callback fn) {
    DtorCallbacks.emplace_back(fn);
    return &DtorCallbacks.back();
  }

  Callback* registerNotificationCallback(Callback fn) {
    NotifCallbacks.emplace_back(fn);
    return &NotifCallbacks.back();
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
    deleteCallback(DtorCallbacks, c);
  }

  void deleteNotificationCallback(Callback* c) {
    deleteCallback(NotifCallbacks, c);
  }

  /// \brief Notifies all listeners (`registerNotificationCallback`
  /// users) of an update.  Assumes the information of the update
  /// is encoded in the state of the derived class, thus only passing
  /// a pointer of type T* to the callback.
  void notify() {
    for (auto callback : NotifCallbacks) {
      callback(reinterpret_cast<T*>(this));
    }
  }

  virtual ~Notifier() {
    for (auto callback : DtorCallbacks) {
      callback(reinterpret_cast<T*>(this));
    }
  }

private:
  std::list<Callback> DtorCallbacks;
  std::list<Callback> NotifCallbacks;
};

#endif /* NOM_SUPPORT_COMMON_H */
