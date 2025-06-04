// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/ATen.h>
#include "minpybind.h"

#ifdef _WIN32
#include <intrin.h>
// https://stackoverflow.com/questions/355967/how-to-use-msvc-intrinsics-to-get-the-equivalent-of-this-gcc-code
inline unsigned int __builtin_clz(unsigned int x) {
    unsigned long r = 0;
    _BitScanReverse(&r, x);
    return (31 - r);
}
#endif

inline int round2min8(int num) {
   int nzeros = __builtin_clz((num - 1)|4);
   return 1 << (32 - nzeros);
}

struct Arena;
template<typename T>
struct OwnedSlice;

template<typename T>
struct Slice {
    Slice()
    :  begin_(nullptr), size_(0), capacity_(0) {}

    template<typename... Args>
    Slice(Arena& arena, Args&&... args);

    T* begin() const {
        return begin_;
    }
    T* end() const {
        return begin_ + size_;
    }
    int size() const {
        return size_;
    }
    int capacity() const {
        return capacity_;
    }

    T& back(int i=-1) {
        return begin_[size_ + i];
    }

    T& operator[](int i) const {
        return begin_[i];
    }
    std::optional<int> index(const T& value) {
        for (int i : enumerate()) {
            if (begin_[i] == value) {
                return i;
            }
        }
        return std::nullopt;
    }
    bool contains(const T& value) {
        return index(value).has_value();
    }

    void insert(Arena& arena, Slice where, Slice to_insert);
    void insert(Arena& arena, Slice where, T v) {
        return insert(arena, where, Slice(&v, &v + 1));
    }
    void insert(Arena& arena, int where, T v) {
        return insert(arena, slice(where, where), v);
    }
    void append(Arena& arena, T value);
    void extend(Arena& arena, Slice to_insert);
    void extend(Arena& arena, const T* begin, const T* end) {
        return extend(arena, Slice<T>((T*)begin, (T*)end));
    }

    bool remove(Arena& A, T value) {
        auto idx = index(value);
        if (idx) {
            insert(A, slice(*idx, *idx + 1), Slice());
        }
        return idx.has_value();
    }

    Slice slice(int begin) {
        return slice(begin, size_);
    }

    Slice slice(int begin, int end) {
        if (begin < 0) {
            begin += size_;
        }
        if (end < 0) {
            end += size_;
        }
        Slice result;
        result.begin_ = begin_ + begin;
        result.size_ = end - begin;
        result.capacity_ = result.size_;
        return result;
    }

    bool inside(Slice where) {
        return begin() <= where.begin() && where.end() <= end();
    }

    irange enumerate() const {
        return irange(size_);
    }

    irange reversed_enumerate() const {
        return irange(size_ - 1, -1, -1);
    }

    bool operator==(const Slice<T>& rhs) const {
        if (size() != rhs.size()) {
            return false;
        }
        return std::equal(begin(), end(), rhs.begin());
    }

    Slice(T* begin, T* end)
    : begin_(begin), size_(end - begin), capacity_(size_) {}

protected:
    static int _length(const T& t) {
        return 1;
    }
    static int _length(Slice t) {
        return t.size_;
    }
    static T* _insert(T*& dst, T t) {
        *dst = std::move(t);
        return ++dst;
    }
    static T* _insert(T*& dst, Slice t) {
        std::memcpy(dst, t.begin_, sizeof(T)*t.size_);
        dst += t.size_;
        return dst;
    }
    T* begin_;
    int size_;
    int capacity_;
    friend struct OwnedSlice<T>;
};

template<typename T>
struct OwnedSlice {
    typedef void (*deleter_t)(Slice<T>);
    static void _no_delete(Slice<T>) {}
    OwnedSlice()
    : deleter_(_no_delete) {}
    OwnedSlice(const OwnedSlice&) = delete;
    OwnedSlice& operator=(const OwnedSlice&) = delete;
    ~OwnedSlice() {
        deleter_(slice_);
        if (slice_.size_ > 8) {
            delete [] slice_.begin_;
        }
    }
    void set(Slice<T> to_own, deleter_t deleter = _no_delete) {
        slice_.size_ = slice_.capacity_ = to_own.size();
        slice_.begin_ = (slice_.size_ > 8) ? new T[slice_.size_] : &small_buf[0];
        std::memcpy(slice_.begin_, to_own.begin(), slice_.size_ * sizeof(T));
        deleter_ = deleter;
    }
    Slice<T> slice() const {
        return slice_;
    }
private:
    Slice<T> slice_;
    deleter_t deleter_;
    T small_buf[8];
};

template<typename T>
inline std::ostream& operator<<(std::ostream& s, const Slice<T>& v) {
    s << "[";
    for (int i : v.enumerate()) {
        if (i > 0) {
            s << ", ";
        }
        s << v[i];
    }
    s << "]";
    return s;
}

struct TensorRef {
    TensorRef()
    : impl_(nullptr){}
    TensorRef(const at::Tensor& t)
    : impl_(t.unsafeGetTensorImpl()) {}
    const at::Tensor& operator*() const {
        return *(at::Tensor*)this;
    }
    at::Tensor* operator->() const {
        return (at::Tensor*)this;
    }
    operator bool() const {
        return impl_ != nullptr;
    }
private:
    at::TensorImpl* impl_;
};

constexpr int ARENA_MAX_SIZE = 4096;
constexpr int ALIGNMENT = 8;
struct Arena {
    Arena()
    : allocated_(0) {}
    template<typename T>
    T* allocate(int n) {
        if (!n) {
            return nullptr;
        }
        int to_allocate = sizeof(T)*n;
        int to_allocate_rounded = ALIGNMENT * ((to_allocate - 1) / ALIGNMENT + 1);
        auto prev_allocated = allocated_;
        allocated_ += to_allocate_rounded;
        if (C10_UNLIKELY_OR_CONST(allocated_ > ARENA_MAX_SIZE)) {
            overflow_.emplace_back(new char[to_allocate]);
            return (T*) &overflow_.back()[0];
        }
        return (T*) (buffer_ + prev_allocated);
    }
    TensorRef autorelease(at::Tensor s) {
        auto ref = TensorRef(s);
        s.unsafeReleaseTensorImpl();
        ar_tensors_.append(*this, ref);
        return ref;
    }
    mpy::handle autorelease(mpy::object obj) {
        ar_objects_.append(*this, obj);
        obj.release();
        return ar_objects_.back();
    }
    ~Arena() {
        for(TensorRef t: ar_tensors_) {
            c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim(t->unsafeGetTensorImpl());
        }
        for(mpy::handle h: ar_objects_) {
            mpy::object::steal(h);
        }
    }
private:
    int64_t allocated_;
    char buffer_[ARENA_MAX_SIZE];
    Slice<TensorRef> ar_tensors_;
    Slice<mpy::handle> ar_objects_;
    std::vector<std::unique_ptr<char[]>> overflow_;
};

template<typename T>
inline void Slice<T>::insert(Arena& arena, Slice where, Slice to_insert) {
    AT_ASSERT(inside(where));
    Slice result = *this;
    /// b------sb---se-----e,  0----n
    T* body_dest = where.begin();
    if (where.size() != to_insert.size()) {
        int new_size = size() - where.size() + to_insert.size();
        T* tail_dest = where.begin() + to_insert.size();
        if (new_size >= capacity_) {
            int new_capacity = new_size ? round2min8(new_size) : 0;
            result.capacity_ = new_capacity;
            result.begin_ = arena.allocate<T>(new_capacity);
            body_dest = result.begin_ + (where.begin() - begin());
            tail_dest = body_dest + to_insert.size();
            //std::memcpy(result.begin_, begin_, sizeof(T)*(where.begin() - begin()));
            std::copy(begin_, begin_ + (where.begin() - begin()), result.begin_);
        }
        std::memmove(tail_dest, where.end(), sizeof(T)*(end() - where.end()));
        result.size_ = new_size;
    }

    //std::memcpy(body_dest, to_insert.begin(), sizeof(T)*to_insert.size());
    std::copy(to_insert.begin(), to_insert.end(), body_dest);
    *this = result;
}

template<typename T>
inline void Slice<T>::append(Arena& arena, T value) {
    Slice result = *this;
    if (size_ == capacity_) {
        int new_size = size_ ? round2min8(size_)*2 : 8;
        T* n = arena.allocate<T>(new_size);
        //memcpy(n, begin_, size_*sizeof(T));
        std::copy(begin_, begin_ + size_, n);
        result.begin_ = n;
        result.capacity_ = new_size;
    }
    result[result.size_++] = std::move(value);
    *this = result;
}

template<typename T>
inline void Slice<T>::extend(Arena& arena, Slice<T> rhs) {
    Slice result = *this;
    result.size_ = size_ + rhs.size();
    if (result.size_ > capacity_) {
        int new_size = round2min8(result.size_);
        T* n = arena.allocate<T>(new_size);
        //memcpy(n, begin_, size_*sizeof(T));
        std::copy(begin_, begin_+size_, n);
        result.begin_ = n;
        result.capacity_ = new_size;
    }
    //memcpy(result.begin_ + size_, rhs.begin(), sizeof(T)*rhs.size());
    std::copy(rhs.begin(), rhs.end(), result.begin_ + size_);
    *this = result;
}

template<typename T>
template<typename... Args>
Slice<T>::Slice(Arena& arena, Args&&... args) {
    int lens[] = {_length(args)...};
    size_ = 0;
    for (auto i : lens) {
        size_ += i;
    }
    capacity_ = size_ ? round2min8(size_) : 0;
    begin_ = arena.allocate<T>(capacity_);
    T* dst_ = begin_;
    T* unused[] = {_insert(dst_, args)...};
    (void) unused;
}
