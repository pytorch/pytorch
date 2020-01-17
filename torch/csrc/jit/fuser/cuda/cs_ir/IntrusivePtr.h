#pragma once
/*
This intrusive ptr was taken from: https://github.com/halide/Halide/blob/master/src/IntrusivePtr.h
 */

#include <atomic>
#include <stdlib.h>

namespace Fuser{

/** A class representing a reference count to be used with IntrusivePtr */
class RefCount {
    std::atomic<int> count;

public:
    RefCount() noexcept
        : count(0) {
    }
    int increment() {
        return ++count;
    }  // Increment and return new value
    int decrement() {
        return --count;
    }  // Decrement and return new value
    bool is_zero() const {
        return count == 0;
    }
};

/**
 * Because in this header we don't yet know how client classes store
 * their RefCount (and we don't want to depend on the declarations of
 * the client classes), any class that you want to hold onto via one
 * of these must provide implementations of ref_count and destroy,
 * which we forward-declare here.
 *
 * E.g. if you want to use IntrusivePtr<MyClass>, then you should
 * define something like this in MyClass.cpp (assuming MyClass has
 * a field: mutable RefCount ref_count):
 *
 * template<> RefCount &ref_count<MyClass>(const MyClass *c) noexcept {return c->ref_count;}
 * template<> void destroy<MyClass>(const MyClass *c) {delete c;}
 */
// @{
template<typename T>
RefCount &ref_count(const T *t) noexcept;
template<typename T>
void destroy(const T *t);
// @}

/** Intrusive shared pointers have a reference count (a
 * RefCount object) stored in the class itself. This is perhaps more
 * efficient than storing it externally, but more importantly, it
 * means it's possible to recover a reference-counted handle from the
 * raw pointer, and it's impossible to have two different reference
 * counts attached to the same raw object. Seeing as we pass around
 * raw pointers to concrete IRNodes and Expr's interchangeably, this
 * is a useful property.
 */
template<typename T>
struct IntrusivePtr {
private:
    void incref(T *p) {
        if (p)
            ref_count(p).increment();
    };

    void decref(T *p) {
        if (p)
            if (ref_count(p).decrement() == 0)
                destroy(p);
    }

protected:
    T *ptr = nullptr;

public:
    T *get() const { return ptr; }

    T &operator*() const { return *ptr; }

    T *operator->() const { return ptr; }

    ~IntrusivePtr() { decref(ptr); }

    IntrusivePtr() = default;

    IntrusivePtr(T *p) : ptr(p) { incref(ptr); }

    IntrusivePtr(const IntrusivePtr<T> &other) noexcept
        : ptr(other.ptr) {
        incref(ptr);
    }

    IntrusivePtr(IntrusivePtr<T> &&other) noexcept
        : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    IntrusivePtr<T> &operator=(const IntrusivePtr<T> &other) {
        if (other.ptr == ptr) return *this;
        // Other can be inside of something owned by this, so we
        // should be careful to incref other before we decref
        // ourselves.
        T *temp = other.ptr;
        incref(temp);
        decref(ptr);
        ptr = temp;
        return *this;
    }

    IntrusivePtr<T> &operator=(IntrusivePtr<T> &&other) noexcept {
        std::swap(ptr, other.ptr);
        return *this;
    }

    /* Handles can be null. This checks that. */
    bool defined() const {
        return ptr != nullptr;
    }

    bool same_as(const IntrusivePtr &other) const {
        return ptr == other.ptr;
    }

    bool operator<(const IntrusivePtr<T> &other) const {
        return ptr < other.ptr;
    }
};

}