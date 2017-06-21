#ifndef CAFFE2_CORE_OBSERVER_H_
#define CAFFE2_CORE_OBSERVER_H_

namespace caffe2 {

/* Use this to implement a Observer using the Observer Pattern template.
 */

template <class T>
class ObserverBase {
 public:
  explicit ObserverBase(T* subject) : subject_(subject) {
    subject_->SetObserver(this);
  }

  virtual bool Start() {
    return false;
  }
  virtual bool Stop() {
    return false;
  }

  virtual ~ObserverBase() noexcept {};

 protected:
  T* subject_;
};

} // namespace

#endif // CAFFE2_CORE_OBSERVER_H_
