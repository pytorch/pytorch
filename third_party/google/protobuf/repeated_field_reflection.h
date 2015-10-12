// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// This header file is protobuf internal. Users should not include this
// file directly.
#ifndef GOOGLE_PROTOBUF_REPEATED_FIELD_REFLECTION_H__
#define GOOGLE_PROTOBUF_REPEATED_FIELD_REFLECTION_H__

#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif

#include <google/protobuf/generated_enum_reflection.h>

namespace google {
namespace protobuf {
namespace internal {
// Interfaces used to implement reflection RepeatedFieldRef API.
// Reflection::GetRepeatedAccessor() should return a pointer to an singleton
// object that implements the below interface.
//
// This interface passes/returns values using void pointers. The actual type
// of the value depends on the field's cpp_type. Following is a mapping from
// cpp_type to the type that should be used in this interface:
//
//   field->cpp_type()      T                Actual type of void*
//   CPPTYPE_INT32        int32                   int32
//   CPPTYPE_UINT32       uint32                  uint32
//   CPPTYPE_INT64        int64                   int64
//   CPPTYPE_UINT64       uint64                  uint64
//   CPPTYPE_DOUBLE       double                  double
//   CPPTYPE_FLOAT        float                   float
//   CPPTYPE_BOOL         bool                    bool
//   CPPTYPE_ENUM         generated enum type     int32
//   CPPTYPE_STRING       string                  string
//   CPPTYPE_MESSAGE      generated message type  google::protobuf::Message
//                        or google::protobuf::Message
//
// Note that for enums we use int32 in the interface.
//
// You can map from T to the actual type using RefTypeTraits:
//   typedef RefTypeTraits<T>::AccessorValueType ActualType;
class LIBPROTOBUF_EXPORT RepeatedFieldAccessor {
 public:
  // Typedefs for clarity.
  typedef void Field;
  typedef void Value;
  typedef void Iterator;

  virtual ~RepeatedFieldAccessor();
  virtual bool IsEmpty(const Field* data) const = 0;
  virtual int Size(const Field* data) const = 0;
  // Depends on the underlying representation of the repeated field, this
  // method can return a pointer to the underlying object if such an object
  // exists, or fill the data into scratch_space and return scratch_space.
  // Callers of this method must ensure scratch_space is a valid pointer
  // to a mutable object of the correct type.
  virtual const Value* Get(
      const Field* data, int index, Value* scratch_space) const = 0;

  virtual void Clear(Field* data) const = 0;
  virtual void Set(Field* data, int index, const Value* value) const = 0;
  virtual void Add(Field* data, const Value* value) const = 0;
  virtual void RemoveLast(Field* data) const = 0;
  virtual void SwapElements(Field* data, int index1, int index2) const = 0;
  virtual void Swap(Field* data, const RepeatedFieldAccessor* other_mutator,
                    Field* other_data) const = 0;

  // Create an iterator that points at the begining of the repeated field.
  virtual Iterator* BeginIterator(const Field* data) const = 0;
  // Create an iterator that points at the end of the repeated field.
  virtual Iterator* EndIterator(const Field* data) const = 0;
  // Make a copy of an iterator and return the new copy.
  virtual Iterator* CopyIterator(const Field* data,
                                 const Iterator* iterator) const = 0;
  // Move an iterator to point to the next element.
  virtual Iterator* AdvanceIterator(const Field* data,
                                    Iterator* iterator) const = 0;
  // Compare whether two iterators point to the same element.
  virtual bool EqualsIterator(const Field* data, const Iterator* a,
                              const Iterator* b) const = 0;
  // Delete an iterator created by BeginIterator(), EndIterator() and
  // CopyIterator().
  virtual void DeleteIterator(const Field* data, Iterator* iterator) const = 0;
  // Like Get() but for iterators.
  virtual const Value* GetIteratorValue(const Field* data,
                                        const Iterator* iterator,
                                        Value* scratch_space) const = 0;

  // Templated methods that make using this interface easier for non-message
  // types.
  template<typename T>
  T Get(const Field* data, int index) const {
    typedef typename RefTypeTraits<T>::AccessorValueType ActualType;
    ActualType scratch_space;
    return static_cast<T>(
        *reinterpret_cast<const ActualType*>(
            Get(data, index, static_cast<Value*>(&scratch_space))));
  }

  template<typename T, typename ValueType>
  void Set(Field* data, int index, const ValueType& value) const {
    typedef typename RefTypeTraits<T>::AccessorValueType ActualType;
    // In this RepeatedFieldAccessor interface we pass/return data using
    // raw pointers. Type of the data these raw pointers point to should
    // be ActualType. Here we have a ValueType object and want a ActualType
    // pointer. We can't cast a ValueType pointer to an ActualType pointer
    // directly because their type might be different (for enums ValueType
    // may be a generated enum type while ActualType is int32). To be safe
    // we make a copy to get a temporary ActualType object and use it.
    ActualType tmp = static_cast<ActualType>(value);
    Set(data, index, static_cast<const Value*>(&tmp));
  }

  template<typename T, typename ValueType>
  void Add(Field* data, const ValueType& value) const {
    typedef typename RefTypeTraits<T>::AccessorValueType ActualType;
    // In this RepeatedFieldAccessor interface we pass/return data using
    // raw pointers. Type of the data these raw pointers point to should
    // be ActualType. Here we have a ValueType object and want a ActualType
    // pointer. We can't cast a ValueType pointer to an ActualType pointer
    // directly because their type might be different (for enums ValueType
    // may be a generated enum type while ActualType is int32). To be safe
    // we make a copy to get a temporary ActualType object and use it.
    ActualType tmp = static_cast<ActualType>(value);
    Add(data, static_cast<const Value*>(&tmp));
  }
};

// Implement (Mutable)RepeatedFieldRef::iterator
template<typename T>
class RepeatedFieldRefIterator
    : public std::iterator<std::forward_iterator_tag, T> {
  typedef typename RefTypeTraits<T>::AccessorValueType AccessorValueType;
  typedef typename RefTypeTraits<T>::IteratorValueType IteratorValueType;
  typedef typename RefTypeTraits<T>::IteratorPointerType IteratorPointerType;

 public:
  // Constructor for non-message fields.
  RepeatedFieldRefIterator(const void* data,
                           const RepeatedFieldAccessor* accessor,
                           bool begin)
      : data_(data), accessor_(accessor),
        iterator_(begin ? accessor->BeginIterator(data) :
                          accessor->EndIterator(data)),
        scratch_space_(new AccessorValueType) {
  }
  // Constructor for message fields.
  RepeatedFieldRefIterator(const void* data,
                           const RepeatedFieldAccessor* accessor,
                           bool begin,
                           AccessorValueType* scratch_space)
      : data_(data), accessor_(accessor),
        iterator_(begin ? accessor->BeginIterator(data) :
                          accessor->EndIterator(data)),
        scratch_space_(scratch_space) {
  }
  ~RepeatedFieldRefIterator() {
    accessor_->DeleteIterator(data_, iterator_);
  }
  RepeatedFieldRefIterator operator++(int) {
    RepeatedFieldRefIterator tmp(*this);
    iterator_ = accessor_->AdvanceIterator(data_, iterator_);
    return tmp;
  }
  RepeatedFieldRefIterator& operator++() {
    iterator_ = accessor_->AdvanceIterator(data_, iterator_);
    return *this;
  }
  IteratorValueType operator*() const {
    return static_cast<IteratorValueType>(
        *static_cast<const AccessorValueType*>(
            accessor_->GetIteratorValue(
                data_, iterator_, scratch_space_.get())));
  }
  IteratorPointerType operator->() const {
    return static_cast<IteratorPointerType>(
        accessor_->GetIteratorValue(
            data_, iterator_, scratch_space_.get()));
  }
  bool operator!=(const RepeatedFieldRefIterator& other) const {
    assert(data_ == other.data_);
    assert(accessor_ == other.accessor_);
    return !accessor_->EqualsIterator(data_, iterator_, other.iterator_);
  }
  bool operator==(const RepeatedFieldRefIterator& other) const {
    return !this->operator!=(other);
  }

  RepeatedFieldRefIterator(const RepeatedFieldRefIterator& other)
      : data_(other.data_), accessor_(other.accessor_),
        iterator_(accessor_->CopyIterator(data_, other.iterator_)) {
  }
  RepeatedFieldRefIterator& operator=(const RepeatedFieldRefIterator& other) {
    if (this != &other) {
      accessor_->DeleteIterator(data_, iterator_);
      data_ = other.data_;
      accessor_ = other.accessor_;
      iterator_ = accessor_->CopyIterator(data_, other.iterator_);
    }
    return *this;
  }

 protected:
  const void* data_;
  const RepeatedFieldAccessor* accessor_;
  void* iterator_;
  google::protobuf::scoped_ptr<AccessorValueType> scratch_space_;
};

// TypeTraits that maps the type parameter T of RepeatedFieldRef or
// MutableRepeatedFieldRef to corresponding iterator type,
// RepeatedFieldAccessor type, etc.
template<typename T>
struct PrimitiveTraits {
  static const bool is_primitive = false;
};
#define DEFINE_PRIMITIVE(TYPE, type) \
    template<> struct PrimitiveTraits<type> { \
      static const bool is_primitive = true; \
      static const FieldDescriptor::CppType cpp_type = \
          FieldDescriptor::CPPTYPE_ ## TYPE; \
    };
DEFINE_PRIMITIVE(INT32, int32)
DEFINE_PRIMITIVE(UINT32, uint32)
DEFINE_PRIMITIVE(INT64, int64)
DEFINE_PRIMITIVE(UINT64, uint64)
DEFINE_PRIMITIVE(FLOAT, float)
DEFINE_PRIMITIVE(DOUBLE, double)
DEFINE_PRIMITIVE(BOOL, bool)
#undef DEFINE_PRIMITIVE

template<typename T>
struct RefTypeTraits<
    T, typename internal::enable_if<PrimitiveTraits<T>::is_primitive>::type> {
  typedef RepeatedFieldRefIterator<T> iterator;
  typedef RepeatedFieldAccessor AccessorType;
  typedef T AccessorValueType;
  typedef T IteratorValueType;
  typedef T* IteratorPointerType;
  static const FieldDescriptor::CppType cpp_type =
      PrimitiveTraits<T>::cpp_type;
  static const Descriptor* GetMessageFieldDescriptor() {
    return NULL;
  }
};

template<typename T>
struct RefTypeTraits<
    T, typename internal::enable_if<is_proto_enum<T>::value>::type> {
  typedef RepeatedFieldRefIterator<T> iterator;
  typedef RepeatedFieldAccessor AccessorType;
  // We use int32 for repeated enums in RepeatedFieldAccessor.
  typedef int32 AccessorValueType;
  typedef T IteratorValueType;
  typedef int32* IteratorPointerType;
  static const FieldDescriptor::CppType cpp_type =
      FieldDescriptor::CPPTYPE_ENUM;
  static const Descriptor* GetMessageFieldDescriptor() {
    return NULL;
  }
};

template<typename T>
struct RefTypeTraits<
    T, typename internal::enable_if<internal::is_same<string, T>::value>::type> {
  typedef RepeatedFieldRefIterator<T> iterator;
  typedef RepeatedFieldAccessor AccessorType;
  typedef string AccessorValueType;
  typedef string IteratorValueType;
  typedef string* IteratorPointerType;
  static const FieldDescriptor::CppType cpp_type =
      FieldDescriptor::CPPTYPE_STRING;
  static const Descriptor* GetMessageFieldDescriptor() {
    return NULL;
  }
};

template<typename T>
struct MessageDescriptorGetter {
  static const Descriptor* get() {
    return T::default_instance().GetDescriptor();
  }
};
template<>
struct MessageDescriptorGetter<Message> {
  static const Descriptor* get() {
    return NULL;
  }
};

template<typename T>
struct RefTypeTraits<
    T, typename internal::enable_if<internal::is_base_of<Message, T>::value>::type> {
  typedef RepeatedFieldRefIterator<T> iterator;
  typedef RepeatedFieldAccessor AccessorType;
  typedef Message AccessorValueType;
  typedef const T& IteratorValueType;
  typedef const T* IteratorPointerType;
  static const FieldDescriptor::CppType cpp_type =
      FieldDescriptor::CPPTYPE_MESSAGE;
  static const Descriptor* GetMessageFieldDescriptor() {
    return MessageDescriptorGetter<T>::get();
  }
};
}  // namespace internal
}  // namespace protobuf
}  // namespace google
#endif  // GOOGLE_PROTOBUF_REPEATED_FIELD_REFLECTION_H__
