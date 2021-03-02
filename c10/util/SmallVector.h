//===- llvm/ADT/SmallVector.h - 'Normally small' vectors --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallVector class.
//
//===----------------------------------------------------------------------===//

// ATen: modified from llvm::SmallVector.
// replaced report_bad_alloc_error with std::bad_alloc
// replaced isPodLike<T> with C10_IS_TRIVIALLY_COPYABLE (moved to Macros.h)
// replaced iterator_range constructor with inline Container&& constructor
// removed LLVM_NODISCARD and LLVM_ATTRIBUTE_ALWAYS_INLINE qualifiers
// removed LLVM_UNLIKELY

#pragma once

#include <c10/util/AlignOf.h>
#include <c10/macros/Macros.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace c10 {

namespace detail {

// From llvm/Support/MathExtras.h
static inline uint64_t NextPowerOf2(uint64_t A) {
  A |= (A >> 1);
  A |= (A >> 2);
  A |= (A >> 4);
  A |= (A >> 8);
  A |= (A >> 16);
  A |= (A >> 32);
  return A + 1;
}

} // namespace detail

/// This is all the non-templated stuff common to all SmallVectors.
class C10_API SmallVectorBase {
 protected:
  void *BeginX, *EndX, *CapacityX;

 protected:
  SmallVectorBase(void* FirstEl, size_t Size)
      : BeginX(FirstEl), EndX(FirstEl), CapacityX((char*)FirstEl + Size) {}

  /// This is an implementation of the grow() method which only works
  /// on POD-like data types and is out of line to reduce code duplication.
  void grow_pod(void* FirstEl, size_t MinSizeInBytes, size_t TSize);

 public:
  /// This returns size()*sizeof(T).
  size_t size_in_bytes() const {
    return size_t((char*)EndX - (char*)BeginX);
  }

  /// capacity_in_bytes - This returns capacity()*sizeof(T).
  size_t capacity_in_bytes() const {
    return size_t((char*)CapacityX - (char*)BeginX);
  }

  bool empty() const {
    return BeginX == EndX;
  }
};

/// This is the part of SmallVectorTemplateBase which does not depend on whether
/// the type T is a POD. The extra dummy template argument is used by ArrayRef
/// to avoid unnecessarily requiring T to be complete.
template <typename T, typename = void>
class SmallVectorTemplateCommon : public SmallVectorBase {
 private:
  template <typename, unsigned>
  friend struct SmallVectorStorage;

  // Allocate raw space for N elements of type T.  If T has a ctor or dtor, we
  // don't want it to be automatically run, so we need to represent the space as
  // something else.  Use an array of char of sufficient alignment.
  using U = AlignedCharArrayUnion<T>;
  U FirstEl;
  // Space after 'FirstEl' is clobbered, do not add any instance vars after it.

 protected:
  SmallVectorTemplateCommon(size_t Size) : SmallVectorBase(&FirstEl, Size) {}

  void grow_pod(size_t MinSizeInBytes, size_t TSize) {
    SmallVectorBase::grow_pod(&FirstEl, MinSizeInBytes, TSize);
  }

  /// Return true if this is a smallvector which has not had dynamic
  /// memory allocated for it.
  bool isSmall() const {
    return BeginX == static_cast<const void*>(&FirstEl);
  }

  /// Put this vector in a state of being small.
  void resetToSmall() {
    BeginX = EndX = CapacityX = &FirstEl;
  }

  void setEnd(T* P) {
    this->EndX = P;
  }

 public:
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = std::reverse_iterator<iterator>;

  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;

  // forward iterator creation methods.
  iterator begin() {
    return (iterator)this->BeginX;
  }
  const_iterator begin() const {
    return (const_iterator)this->BeginX;
  }
  iterator end() {
    return (iterator)this->EndX;
  }
  const_iterator end() const {
    return (const_iterator)this->EndX;
  }

 protected:
  iterator capacity_ptr() {
    return (iterator)this->CapacityX;
  }
  const_iterator capacity_ptr() const {
    return (const_iterator)this->CapacityX;
  }

 public:
  // reverse iterator creation methods.
  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() {
    return reverse_iterator(begin());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  size_type size() const {
    return end() - begin();
  }
  size_type max_size() const {
    return size_type(-1) / sizeof(T);
  }

  /// Return the total number of elements in the currently allocated buffer.
  size_t capacity() const {
    return capacity_ptr() - begin();
  }

  /// Return a pointer to the vector's buffer, even if empty().
  pointer data() {
    return pointer(begin());
  }
  /// Return a pointer to the vector's buffer, even if empty().
  const_pointer data() const {
    return const_pointer(begin());
  }

  // SmallVector::at is NOT from LLVM.
  reference at(size_type idx) {
    assert(idx < size());
    return begin()[idx];
  }
  const_reference at(size_type idx) const {
    assert(idx < size());
    return begin()[idx];
  }

  reference operator[](size_type idx) {
    assert(idx < size());
    return begin()[idx];
  }
  const_reference operator[](size_type idx) const {
    assert(idx < size());
    return begin()[idx];
  }

  reference front() {
    assert(!empty());
    return begin()[0];
  }
  const_reference front() const {
    assert(!empty());
    return begin()[0];
  }

  reference back() {
    assert(!empty());
    return end()[-1];
  }
  const_reference back() const {
    assert(!empty());
    return end()[-1];
  }
};

/// SmallVectorTemplateBase<isPodLike = false> - This is where we put method
/// implementations that are designed to work with non-POD-like T's.
template <typename T, bool isPodLike>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {
 protected:
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  static void destroy_range(T* S, T* E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

  /// Move the range [Iit, Eit) into the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template <typename It1, typename It2>
  static void uninitialized_move(It1 Iit, It1 Eit, It2 Dest) {
    std::uninitialized_copy(
        std::make_move_iterator(Iit), std::make_move_iterator(Eit), Dest);
  }

  /// Copy the range [Iit, Eit) onto the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template <typename It1, typename It2>
  static void uninitialized_copy(It1 Iit, It1 Eit, It2 Dest) {
    std::uninitialized_copy(Iit, Eit, Dest);
  }

  /// Grow the allocated memory (without initializing new elements), doubling
  /// the size of the allocated memory. Guarantees space for at least one more
  /// element, or MinSize more elements if specified.
  void grow(size_t MinSize = 0);

 public:
  void push_back(const T& Elt) {
    if (this->EndX >= this->CapacityX)
      this->grow();
    ::new ((void*)this->end()) T(Elt);
    this->setEnd(this->end() + 1);
  }

  void push_back(T&& Elt) {
    if (this->EndX >= this->CapacityX)
      this->grow();
    ::new ((void*)this->end()) T(::std::move(Elt));
    this->setEnd(this->end() + 1);
  }

  void pop_back() {
    this->setEnd(this->end() - 1);
    this->end()->~T();
  }
};

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool isPodLike>
void SmallVectorTemplateBase<T, isPodLike>::grow(size_t MinSize) {
  size_t CurCapacity = this->capacity();
  size_t CurSize = this->size();
  // Always grow, even from zero.
  size_t NewCapacity = size_t(detail::NextPowerOf2(CurCapacity + 2));
  if (NewCapacity < MinSize)
    NewCapacity = MinSize;
  T* NewElts = static_cast<T*>(malloc(NewCapacity * sizeof(T)));
  if (NewElts == nullptr)
    throw std::bad_alloc();

  // Move the elements over.
  this->uninitialized_move(this->begin(), this->end(), NewElts);

  // Destroy the original elements.
  destroy_range(this->begin(), this->end());

  // If this wasn't grown from the inline copy, deallocate the old space.
  if (!this->isSmall())
    free(this->begin());

  this->setEnd(NewElts + CurSize);
  this->BeginX = NewElts;
  this->CapacityX = this->begin() + NewCapacity;
}

/// SmallVectorTemplateBase<isPodLike = true> - This is where we put method
/// implementations that are designed to work with POD-like T's.
template <typename T>
class SmallVectorTemplateBase<T, true> : public SmallVectorTemplateCommon<T> {
 protected:
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  // No need to do a destroy loop for POD's.
  static void destroy_range(T*, T*) {}

  /// Move the range [Iit, Eit) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename It1, typename It2>
  static void uninitialized_move(It1 Iit, It1 Eit, It2 Dest) {
    // Just do a copy.
    uninitialized_copy(Iit, Eit, Dest);
  }

  /// Copy the range [Iit, Eit) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename It1, typename It2>
  static void uninitialized_copy(It1 Iit, It1 Eit, It2 Dest) {
    // Arbitrary iterator types; just use the basic implementation.
    std::uninitialized_copy(Iit, Eit, Dest);
  }

  /// Copy the range [Iit, Eit) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename T1, typename T2>
  static void uninitialized_copy(
      T1* Iit,
      T1* Eit,
      T2* Dest,
      typename std::enable_if<
          std::is_same<typename std::remove_const<T1>::type, T2>::value>::
          type* = nullptr) {
    // Use memcpy for PODs iterated by pointers (which includes SmallVector
    // iterators): std::uninitialized_copy optimizes to memmove, but we can
    // use memcpy here. Note that Iit and Eit are iterators and thus might be
    // invalid for memcpy if they are equal.
    if (Iit != Eit)
      memcpy(Dest, Iit, (Eit - Iit) * sizeof(T));
  }

  /// Double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(size_t MinSize = 0) {
    this->grow_pod(MinSize * sizeof(T), sizeof(T));
  }

 public:
  void push_back(const T& Elt) {
    if (this->EndX >= this->CapacityX)
      this->grow();
    memcpy(this->end(), &Elt, sizeof(T));
    this->setEnd(this->end() + 1);
  }

  void pop_back() {
    this->setEnd(this->end() - 1);
  }
};

/// This class consists of common code factored out of the SmallVector class to
/// reduce code duplication based on the SmallVector 'N' template parameter.
/// Warning: C10_IS_TRIVIALLY_COPYABLE may not always detect non-POD
/// type correctly. For example, std::unique_ptr may be treated as POD and cause
/// memory leaks.
template <typename T>
class SmallVectorImpl
    : public SmallVectorTemplateBase<T, C10_IS_TRIVIALLY_COPYABLE(T)> {
  using SuperClass = SmallVectorTemplateBase<T, C10_IS_TRIVIALLY_COPYABLE(T)>;

 public:
  using iterator = typename SuperClass::iterator;
  using const_iterator = typename SuperClass::const_iterator;
  using size_type = typename SuperClass::size_type;

 protected:
  // Default ctor - Initialize to empty.
  explicit SmallVectorImpl(unsigned N)
      : SmallVectorTemplateBase<T, C10_IS_TRIVIALLY_COPYABLE(T)>(N * sizeof(T)) {
  }

 public:
  SmallVectorImpl(const SmallVectorImpl&) = delete;

  ~SmallVectorImpl() {
    // Destroy the constructed elements in the vector.
    this->destroy_range(this->begin(), this->end());

    // If this wasn't grown from the inline copy, deallocate the old space.
    if (!this->isSmall())
      free(this->begin());
  }

  void clear() {
    this->destroy_range(this->begin(), this->end());
    this->EndX = this->BeginX;
  }

  void resize(size_type N) {
    if (N < this->size()) {
      this->destroy_range(this->begin() + N, this->end());
      this->setEnd(this->begin() + N);
    } else if (N > this->size()) {
      if (this->capacity() < N)
        this->grow(N);
      auto Iit = this->end();
      for (auto Eit = this->begin() + N; Iit != Eit; ++Iit)
        new (&*Iit) T();
      this->setEnd(this->begin() + N);
    }
  }

  void resize(size_type N, const T& NV) {
    if (N < this->size()) {
      this->destroy_range(this->begin() + N, this->end());
      this->setEnd(this->begin() + N);
    } else if (N > this->size()) {
      if (this->capacity() < N)
        this->grow(N);
      std::uninitialized_fill(this->end(), this->begin() + N, NV);
      this->setEnd(this->begin() + N);
    }
  }

  void reserve(size_type N) {
    if (this->capacity() < N)
      this->grow(N);
  }

  T pop_back_val() {
    T Result = ::std::move(this->back());
    this->pop_back();
    return Result;
  }

  void swap(SmallVectorImpl& RHS);

  /// Add the specified range to the end of the SmallVector.
  template <
      typename in_iter,
      typename = typename std::enable_if<std::is_convertible<
          typename std::iterator_traits<in_iter>::iterator_category,
          std::input_iterator_tag>::value>::type>
  void append(in_iter in_start, in_iter in_end) {
    size_type NumInputs = std::distance(in_start, in_end);
    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr() - this->end()))
      this->grow(this->size() + NumInputs);

    // Copy the new elements over.
    this->uninitialized_copy(in_start, in_end, this->end());
    this->setEnd(this->end() + NumInputs);
  }

  /// Add the specified range to the end of the SmallVector.
  void append(size_type NumInputs, const T& Elt) {
    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr() - this->end()))
      this->grow(this->size() + NumInputs);

    // Copy the new elements over.
    std::uninitialized_fill_n(this->end(), NumInputs, Elt);
    this->setEnd(this->end() + NumInputs);
  }

  void append(std::initializer_list<T> IL) {
    append(IL.begin(), IL.end());
  }

  // FIXME: Consider assigning over existing elements, rather than clearing &
  // re-initializing them - for all assign(...) variants.

  void assign(size_type NumElts, const T& Elt) {
    clear();
    if (this->capacity() < NumElts)
      this->grow(NumElts);
    this->setEnd(this->begin() + NumElts);
    std::uninitialized_fill(this->begin(), this->end(), Elt);
  }

  template <
      typename in_iter,
      typename = typename std::enable_if<std::is_convertible<
          typename std::iterator_traits<in_iter>::iterator_category,
          std::input_iterator_tag>::value>::type>
  void assign(in_iter in_start, in_iter in_end) {
    clear();
    append(in_start, in_end);
  }

  void assign(std::initializer_list<T> IL) {
    clear();
    append(IL);
  }

  iterator erase(const_iterator CIit) {
    // Just cast away constness because this is a non-const member function.
    iterator Iit = const_cast<iterator>(CIit);

    assert(Iit >= this->begin() && "Iterator to erase is out of bounds.");
    assert(Iit < this->end() && "Erasing at past-the-end iterator.");

    iterator Nit = Iit;
    // Shift all elts down one.
    std::move(Iit + 1, this->end(), Iit);
    // Drop the last elt.
    this->pop_back();
    return (Nit);
  }

  iterator erase(const_iterator CSit, const_iterator CEit) {
    // Just cast away constness because this is a non-const member function.
    iterator Sit = const_cast<iterator>(CSit);
    iterator Eit = const_cast<iterator>(CEit);

    assert(Sit >= this->begin() && "Range to erase is out of bounds.");
    assert(Sit <= Eit && "Trying to erase invalid range.");
    assert(Eit <= this->end() && "Trying to erase past the end.");

    iterator Nit = Sit;
    // Shift all elts down.
    iterator Iit = std::move(Eit, this->end(), Sit);
    // Drop the last elts.
    this->destroy_range(Iit, this->end());
    this->setEnd(Iit);
    return (Nit);
  }

  iterator insert(iterator Iit, T&& Elt) {
    if (Iit == this->end()) { // Important special case for empty vector.
      this->push_back(::std::move(Elt));
      return this->end() - 1;
    }

    assert(Iit >= this->begin() && "Insertion iterator is out of bounds.");
    assert(Iit <= this->end() && "Inserting past the end of the vector.");

    if (this->EndX >= this->CapacityX) {
      size_t EltNo = Iit - this->begin();
      this->grow();
      Iit = this->begin() + EltNo;
    }

    ::new ((void*)this->end()) T(::std::move(this->back()));
    // Push everything else over.
    std::move_backward(Iit, this->end() - 1, this->end());
    this->setEnd(this->end() + 1);

    // If we just moved the element we're inserting, be sure to update
    // the reference.
    T* EltPtr = &Elt;
    if (Iit <= EltPtr && EltPtr < this->EndX)
      ++EltPtr;

    *Iit = ::std::move(*EltPtr);
    return Iit;
  }

  iterator insert(iterator Iit, const T& Elt) {
    if (Iit == this->end()) { // Important special case for empty vector.
      this->push_back(Elt);
      return this->end() - 1;
    }

    assert(Iit >= this->begin() && "Insertion iterator is out of bounds.");
    assert(Iit <= this->end() && "Inserting past the end of the vector.");

    if (this->EndX >= this->CapacityX) {
      size_t EltNo = Iit - this->begin();
      this->grow();
      Iit = this->begin() + EltNo;
    }
    ::new ((void*)this->end()) T(std::move(this->back()));
    // Push everything else over.
    std::move_backward(Iit, this->end() - 1, this->end());
    this->setEnd(this->end() + 1);

    // If we just moved the element we're inserting, be sure to update
    // the reference.
    const T* EltPtr = &Elt;
    if (Iit <= EltPtr && EltPtr < this->EndX)
      ++EltPtr;

    *Iit = *EltPtr;
    return Iit;
  }

  iterator insert(iterator Iit, size_type NumToInsert, const T& Elt) {
    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = Iit - this->begin();

    if (Iit == this->end()) { // Important special case for empty vector.
      append(NumToInsert, Elt);
      return this->begin() + InsertElt;
    }

    assert(Iit >= this->begin() && "Insertion iterator is out of bounds.");
    assert(Iit <= this->end() && "Inserting past the end of the vector.");

    // Ensure there is enough space.
    reserve(this->size() + NumToInsert);

    // Uninvalidate the iterator.
    Iit = this->begin() + InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end() - Iit) >= NumToInsert) {
      T* OldEnd = this->end();
      append(
          std::move_iterator<iterator>(this->end() - NumToInsert),
          std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      std::move_backward(Iit, OldEnd - NumToInsert, OldEnd);

      std::fill_n(Iit, NumToInsert, Elt);
      return Iit;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T* OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd - Iit;
    this->uninitialized_move(Iit, OldEnd, this->end() - NumOverwritten);

    // Replace the overwritten part.
    std::fill_n(Iit, NumOverwritten, Elt);

    // Insert the non-overwritten middle part.
    std::uninitialized_fill_n(OldEnd, NumToInsert - NumOverwritten, Elt);
    return Iit;
  }

  template <
      typename ItTy,
      typename = typename std::enable_if<std::is_convertible<
          typename std::iterator_traits<ItTy>::iterator_category,
          std::input_iterator_tag>::value>::type>
  iterator insert(iterator Iit, ItTy From, ItTy To) {
    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = Iit - this->begin();

    if (Iit == this->end()) { // Important special case for empty vector.
      append(From, To);
      return this->begin() + InsertElt;
    }

    assert(Iit >= this->begin() && "Insertion iterator is out of bounds.");
    assert(Iit <= this->end() && "Inserting past the end of the vector.");

    size_t NumToInsert = std::distance(From, To);

    // Ensure there is enough space.
    reserve(this->size() + NumToInsert);

    // Uninvalidate the iterator.
    Iit = this->begin() + InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end() - Iit) >= NumToInsert) {
      T* OldEnd = this->end();
      append(
          std::move_iterator<iterator>(this->end() - NumToInsert),
          std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      std::move_backward(Iit, OldEnd - NumToInsert, OldEnd);

      std::copy(From, To, Iit);
      return Iit;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T* OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd - Iit;
    this->uninitialized_move(Iit, OldEnd, this->end() - NumOverwritten);

    // Replace the overwritten part.
    for (T* J = Iit; NumOverwritten > 0; --NumOverwritten) {
      *J = *From;
      ++J;
      ++From;
    }

    // Insert the non-overwritten middle part.
    this->uninitialized_copy(From, To, OldEnd);
    return Iit;
  }

  void insert(iterator Iit, std::initializer_list<T> IL) {
    insert(Iit, IL.begin(), IL.end());
  }

  template <typename... ArgTypes>
  void emplace_back(ArgTypes&&... Args) {
    if (this->EndX >= this->CapacityX)
      this->grow();
    ::new ((void*)this->end()) T(std::forward<ArgTypes>(Args)...);
    this->setEnd(this->end() + 1);
  }

  SmallVectorImpl& operator=(const SmallVectorImpl& RHS);

  SmallVectorImpl& operator=(SmallVectorImpl&& RHS);

  bool operator==(const SmallVectorImpl& RHS) const {
    if (this->size() != RHS.size())
      return false;
    return std::equal(this->begin(), this->end(), RHS.begin());
  }
  bool operator!=(const SmallVectorImpl& RHS) const {
    return !(*this == RHS);
  }

  bool operator<(const SmallVectorImpl& RHS) const {
    return std::lexicographical_compare(
        this->begin(), this->end(), RHS.begin(), RHS.end());
  }

  /// Set the array size to \p N, which the current array must have enough
  /// capacity for.
  ///
  /// This does not construct or destroy any elements in the vector.
  ///
  /// Clients can use this in conjunction with capacity() to write past the end
  /// of the buffer when they know that more elements are available, and only
  /// update the size later. This avoids the cost of value initializing elements
  /// which will only be overwritten.
  void set_size(size_type N) {
    assert(N <= this->capacity());
    this->setEnd(this->begin() + N);
  }
};

template <typename T>
void SmallVectorImpl<T>::swap(SmallVectorImpl<T>& RHS) {
  if (this == &RHS)
    return;

  // We can only avoid copying elements if neither vector is small.
  if (!this->isSmall() && !RHS.isSmall()) {
    std::swap(this->BeginX, RHS.BeginX);
    std::swap(this->EndX, RHS.EndX);
    std::swap(this->CapacityX, RHS.CapacityX);
    return;
  }
  if (RHS.size() > this->capacity())
    this->grow(RHS.size());
  if (this->size() > RHS.capacity())
    RHS.grow(this->size());

  // Swap the shared elements.
  size_t NumShared = this->size();
  if (NumShared > RHS.size())
    NumShared = RHS.size();
  for (size_type i = 0; i != NumShared; ++i)
    std::swap((*this)[i], RHS[i]);

  // Copy over the extra elts.
  if (this->size() > RHS.size()) {
    size_t EltDiff = this->size() - RHS.size();
    this->uninitialized_copy(this->begin() + NumShared, this->end(), RHS.end());
    RHS.setEnd(RHS.end() + EltDiff);
    this->destroy_range(this->begin() + NumShared, this->end());
    this->setEnd(this->begin() + NumShared);
  } else if (RHS.size() > this->size()) {
    size_t EltDiff = RHS.size() - this->size();
    this->uninitialized_copy(RHS.begin() + NumShared, RHS.end(), this->end());
    this->setEnd(this->end() + EltDiff);
    this->destroy_range(RHS.begin() + NumShared, RHS.end());
    RHS.setEnd(RHS.begin() + NumShared);
  }
}

template <typename T>
SmallVectorImpl<T>& SmallVectorImpl<T>::operator=(
    const SmallVectorImpl<T>& RHS) {
  // Avoid self-assignment.
  if (this == &RHS)
    return *this;

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd;
    if (RHSSize)
      NewEnd = std::copy(RHS.begin(), RHS.begin() + RHSSize, this->begin());
    else
      NewEnd = this->begin();

    // Destroy excess elements.
    this->destroy_range(NewEnd, this->end());

    // Trim.
    this->setEnd(NewEnd);
    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: don't do this if they're efficiently movable.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->destroy_range(this->begin(), this->end());
    this->setEnd(this->begin());
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    std::copy(RHS.begin(), RHS.begin() + CurSize, this->begin());
  }

  // Copy construct the new elements in place.
  this->uninitialized_copy(
      RHS.begin() + CurSize, RHS.end(), this->begin() + CurSize);

  // Set end.
  this->setEnd(this->begin() + RHSSize);
  return *this;
}

template <typename T>
SmallVectorImpl<T>& SmallVectorImpl<T>::operator=(SmallVectorImpl<T>&& RHS) {
  // Avoid self-assignment.
  if (this == &RHS)
    return *this;

  // If the RHS isn't small, clear this vector and then steal its buffer.
  if (!RHS.isSmall()) {
    this->destroy_range(this->begin(), this->end());
    if (!this->isSmall())
      free(this->begin());
    this->BeginX = RHS.BeginX;
    this->EndX = RHS.EndX;
    this->CapacityX = RHS.CapacityX;
    RHS.resetToSmall();
    return *this;
  }

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd = this->begin();
    if (RHSSize)
      NewEnd = std::move(RHS.begin(), RHS.end(), NewEnd);

    // Destroy excess elements and trim the bounds.
    this->destroy_range(NewEnd, this->end());
    this->setEnd(NewEnd);

    // Clear the RHS.
    RHS.clear();

    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: this may not actually make any sense if we can efficiently move
  // elements.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->destroy_range(this->begin(), this->end());
    this->setEnd(this->begin());
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    std::move(RHS.begin(), RHS.begin() + CurSize, this->begin());
  }

  // Move-construct the new elements in place.
  this->uninitialized_move(
      RHS.begin() + CurSize, RHS.end(), this->begin() + CurSize);

  // Set end.
  this->setEnd(this->begin() + RHSSize);

  RHS.clear();
  return *this;
}

/// Storage for the SmallVector elements which aren't contained in
/// SmallVectorTemplateCommon. There are 'N-1' elements here. The remaining '1'
/// element is in the base class. This is specialized for the N=1 and N=0 cases
/// to avoid allocating unnecessary storage.
template <typename T, unsigned N>
struct SmallVectorStorage {
  typename SmallVectorTemplateCommon<T>::U InlineElts[N - 1];
};
template <typename T>
struct SmallVectorStorage<T, 1> {};
template <typename T>
struct SmallVectorStorage<T, 0> {};

/// This is a 'vector' (really, a variable-sized array), optimized
/// for the case when the array is small.  It contains some number of elements
/// in-place, which allows it to avoid heap allocation when the actual number of
/// elements is below that threshold.  This allows normal "small" cases to be
/// fast without losing generality for large inputs.
///
/// Note that this does not attempt to be exception safe.
///
template <typename T, unsigned N>
class SmallVector : public SmallVectorImpl<T> {
  /// Inline space for elements which aren't stored in the base class.
  SmallVectorStorage<T, N> Storage;

 public:
  SmallVector() : SmallVectorImpl<T>(N) {}

  explicit SmallVector(size_t Size, const T& Value = T())
      : SmallVectorImpl<T>(N) {
    this->assign(Size, Value);
  }

  template <
      typename ItTy,
      typename = typename std::enable_if<std::is_convertible<
          typename std::iterator_traits<ItTy>::iterator_category,
          std::input_iterator_tag>::value>::type>
  SmallVector(ItTy S, ItTy E) : SmallVectorImpl<T>(N) {
    this->append(S, E);
  }

  // note: The enable_if restricts Container to types that have a .begin() and .end()
  // that return valid input iterators.
  template <typename Container, std::enable_if_t<
      std::is_convertible<
          typename std::iterator_traits<decltype(std::declval<Container>().begin())>::iterator_category,
          std::input_iterator_tag
      >::value &&
      std::is_convertible<
          typename std::iterator_traits<decltype(std::declval<Container>().end())>::iterator_category,
          std::input_iterator_tag
      >::value, int> = 0>
  explicit SmallVector(Container&& c) : SmallVectorImpl<T>(N) {
    this->append(c.begin(), c.end());
  }

  SmallVector(std::initializer_list<T> IL) : SmallVectorImpl<T>(N) {
    this->assign(IL);
  }

  SmallVector(const SmallVector& RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(RHS);
  }

  const SmallVector& operator=(const SmallVector& RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }

  SmallVector(SmallVector&& RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  // note: The enable_if restricts Container to types that have a .begin() and .end()
  // that return valid input iterators.
  template <typename Container, std::enable_if_t<
      std::is_convertible<
          typename std::iterator_traits<decltype(std::declval<Container>().begin())>::iterator_category,
          std::input_iterator_tag
      >::value &&
      std::is_convertible<
          typename std::iterator_traits<decltype(std::declval<Container>().end())>::iterator_category,
          std::input_iterator_tag
      >::value, int> = 0>
  const SmallVector& operator=(const Container& RHS) {
    this->assign(RHS.begin(), RHS.end());
    return *this;
  }

  SmallVector(SmallVectorImpl<T>&& RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  const SmallVector& operator=(SmallVector&& RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  const SmallVector& operator=(SmallVectorImpl<T>&& RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  // note: The enable_if restricts Container to types that have a .begin() and .end()
  // that return valid input iterators.
  template <typename Container, std::enable_if_t<
      std::is_convertible<
          typename std::iterator_traits<decltype(std::declval<Container>().begin())>::iterator_category,
          std::input_iterator_tag
      >::value &&
      std::is_convertible<
          typename std::iterator_traits<decltype(std::declval<Container>().end())>::iterator_category,
          std::input_iterator_tag
      >::value, int> = 0>
  const SmallVector& operator=(Container&& C) {
    this->assign(C.begin(), C.end());
    return *this;
  }

  const SmallVector& operator=(std::initializer_list<T> IL) {
    this->assign(IL);
    return *this;
  }
};

template <typename T, unsigned N>
inline size_t capacity_in_bytes(const SmallVector<T, N>& X) {
  return X.capacity_in_bytes();
}

template <typename T, unsigned N>
std::ostream& operator<<(std::ostream & out, const SmallVector<T, N>& list) {
  int i = 0;
  out << "[";
  for(auto e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "]";
  return out;
}

} // end namespace c10

namespace std {

/// Implement std::swap in terms of SmallVector swap.
template <typename T>
inline void swap(c10::SmallVectorImpl<T>& LHS, c10::SmallVectorImpl<T>& RHS) {
  LHS.swap(RHS);
}

/// Implement std::swap in terms of SmallVector swap.
template <typename T, unsigned N>
inline void swap(c10::SmallVector<T, N>& LHS, c10::SmallVector<T, N>& RHS) {
  LHS.swap(RHS);
}

} // end namespace std
