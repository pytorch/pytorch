#pragma once

//simple intrusive pointer implementation, cleaned up from boost intrusive_ptr
namespace torch {
template<typename T>
class IntrusivePtr {
public:
  IntrusivePtr()
  : ptr(nullptr){}
  explicit IntrusivePtr(T * self)
  : ptr(self) {
    if(ptr != nullptr)
      ptr->retain();
  }
  IntrusivePtr(IntrusivePtr const & rhs)
  : ptr(rhs.ptr) {
    if(ptr != nullptr)
      ptr->retain();
  }
  IntrusivePtr(IntrusivePtr && rhs)
  : ptr(rhs.ptr) {
    rhs.ptr = nullptr;
  }
  ~IntrusivePtr() {
    if(ptr != nullptr)
      ptr->release();
  }
  IntrusivePtr & operator=(IntrusivePtr && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  IntrusivePtr & operator=(IntrusivePtr const & rhs) & {
      //IntrusivePtr ctor retains original rhs.ptr
      //then rhs.ptr is swapped with this->ptr
      //finally IntrusivePtr dtor releases rhs.ptr, which was originally this->ptr
      IntrusivePtr(rhs).swap(*this);
      return *this;
  }
  T * get() {
    return ptr;
  }
  void swap(IntrusivePtr & rhs) {
    T * tmp = ptr;
    ptr = rhs.ptr;
    rhs.ptr = tmp;
  }
  T & operator*() const {
    return *ptr;
  }
  T * operator->() const {
    return ptr;
  }
  operator bool() const {
    return ptr != nullptr;
  }
private:
  T * ptr;
};

}
