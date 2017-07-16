#pragma once

namespace caffe2 {

/**
 *  Use this to implement a Observer using the Observer Pattern template.
 */

template <class T>
class ObserverBase {
 public:
  explicit ObserverBase(T* subject) : subject_(subject) {}

  virtual bool Start() {
    return false;
  }
  virtual bool Stop() {
    return false;
  }

  virtual ~ObserverBase() noexcept {};

  T* subject() const {
    return subject_;
  }

 protected:
  T* subject_;
};

} // namespace caffe2
