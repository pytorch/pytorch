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

#ifndef GOOGLE_PROTOBUF_MAP_H__
#define GOOGLE_PROTOBUF_MAP_H__

#include <iterator>
#include <google/protobuf/stubs/hash.h>
#include <limits>  // To support Visual Studio 2008

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/generated_enum_util.h>
#include <google/protobuf/map_type_handler.h>
#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>

namespace google {
namespace protobuf {

template <typename Key, typename T>
class Map;

template <typename Enum> struct is_proto_enum;

class MapIterator;

namespace internal {
template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
class MapFieldLite;

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
class MapField;

template <typename Key, typename T>
class TypeDefinedMapFieldBase;

class DynamicMapField;

class GeneratedMessageReflection;
}  // namespace internal

#define TYPE_CHECK(EXPECTEDTYPE, METHOD)                                \
  if (type() != EXPECTEDTYPE) {                                         \
    GOOGLE_LOG(FATAL)                                                          \
        << "Protocol Buffer map usage error:\n"                         \
        << METHOD << " type does not match\n"                           \
        << "  Expected : "                                              \
        << FieldDescriptor::CppTypeName(EXPECTEDTYPE) << "\n"           \
        << "  Actual   : "                                              \
        << FieldDescriptor::CppTypeName(type());                        \
  }

// MapKey is an union type for representing any possible
// map key.
class LIBPROTOBUF_EXPORT MapKey {
 public:
  MapKey() : type_(0) {
  }
  MapKey(const MapKey& other) : type_(0) {
    CopyFrom(other);
  }

  ~MapKey() {
    if (type_ == FieldDescriptor::CPPTYPE_STRING) {
      delete val_.string_value_;
    }
  }

  FieldDescriptor::CppType type() const {
    if (type_ == 0) {
      GOOGLE_LOG(FATAL)
          << "Protocol Buffer map usage error:\n"
          << "MapKey::type MapKey is not initialized. "
          << "Call set methods to initialize MapKey.";
    }
    return (FieldDescriptor::CppType)type_;
  }

  void SetInt64Value(int64 value) {
    SetType(FieldDescriptor::CPPTYPE_INT64);
    val_.int64_value_ = value;
  }
  void SetUInt64Value(uint64 value) {
    SetType(FieldDescriptor::CPPTYPE_UINT64);
    val_.uint64_value_ = value;
  }
  void SetInt32Value(int32 value) {
    SetType(FieldDescriptor::CPPTYPE_INT32);
    val_.int32_value_ = value;
  }
  void SetUInt32Value(uint32 value) {
    SetType(FieldDescriptor::CPPTYPE_UINT32);
    val_.uint32_value_ = value;
  }
  void SetBoolValue(bool value) {
    SetType(FieldDescriptor::CPPTYPE_BOOL);
    val_.bool_value_ = value;
  }
  void SetStringValue(const string& val) {
    SetType(FieldDescriptor::CPPTYPE_STRING);
    *val_.string_value_ = val;
  }

  int64 GetInt64Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT64,
               "MapKey::GetInt64Value");
    return val_.int64_value_;
  }
  uint64 GetUInt64Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT64,
               "MapKey::GetUInt64Value");
    return val_.uint64_value_;
  }
  int32 GetInt32Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT32,
               "MapKey::GetInt32Value");
    return val_.int32_value_;
  }
  uint32 GetUInt32Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT32,
               "MapKey::GetUInt32Value");
    return val_.uint32_value_;
  }
  int32 GetBoolValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_BOOL,
               "MapKey::GetBoolValue");
    return val_.bool_value_;
  }
  const string& GetStringValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_STRING,
               "MapKey::GetStringValue");
    return *val_.string_value_;
  }

  bool operator==(const MapKey& other) const {
    if (type_ != other.type_) {
      return false;
    }
    switch (type()) {
      case FieldDescriptor::CPPTYPE_STRING:
        return *val_.string_value_ == *other.val_.string_value_;
      case FieldDescriptor::CPPTYPE_INT64:
        return val_.int64_value_ == other.val_.int64_value_;
      case FieldDescriptor::CPPTYPE_INT32:
        return val_.int32_value_ == other.val_.int32_value_;
      case FieldDescriptor::CPPTYPE_UINT64:
        return val_.uint64_value_ == other.val_.uint64_value_;
      case FieldDescriptor::CPPTYPE_UINT32:
        return val_.uint32_value_ == other.val_.uint32_value_;
      case FieldDescriptor::CPPTYPE_BOOL:
        return val_.bool_value_ == other.val_.bool_value_;
      case FieldDescriptor::CPPTYPE_DOUBLE:
      case FieldDescriptor::CPPTYPE_FLOAT:
      case FieldDescriptor::CPPTYPE_ENUM:
      case FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Can't get here.";
        return false;
    }
  }

  void CopyFrom(const MapKey& other) {
    SetType(other.type());
    switch (type_) {
      case FieldDescriptor::CPPTYPE_STRING:
        *val_.string_value_ = *other.val_.string_value_;
        break;
      case FieldDescriptor::CPPTYPE_INT64:
        val_.int64_value_ = other.val_.int64_value_;
        break;
      case FieldDescriptor::CPPTYPE_INT32:
        val_.int32_value_ = other.val_.int32_value_;
        break;
      case FieldDescriptor::CPPTYPE_UINT64:
        val_.uint64_value_ = other.val_.uint64_value_;
        break;
      case FieldDescriptor::CPPTYPE_UINT32:
        val_.uint32_value_ = other.val_.uint32_value_;
        break;
      case FieldDescriptor::CPPTYPE_BOOL:
        val_.bool_value_ = other.val_.bool_value_;
        break;
      case FieldDescriptor::CPPTYPE_DOUBLE:
      case FieldDescriptor::CPPTYPE_FLOAT:
      case FieldDescriptor::CPPTYPE_ENUM:
      case FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Can't get here.";
        break;
    }
  }

 private:
  template <typename K, typename V>
  friend class internal::TypeDefinedMapFieldBase;
  friend class MapIterator;
  friend class internal::DynamicMapField;

  union KeyValue {
    KeyValue() {}
    string* string_value_;
    int64 int64_value_;
    int32 int32_value_;
    uint64 uint64_value_;
    uint32 uint32_value_;
    bool bool_value_;
  } val_;

  void SetType(FieldDescriptor::CppType type) {
    if (type_ == type) return;
    if (type_ == FieldDescriptor::CPPTYPE_STRING) {
      delete val_.string_value_;
    }
    type_ = type;
    if (type_ == FieldDescriptor::CPPTYPE_STRING) {
      val_.string_value_ = new string;
    }
  }

  // type_ is 0 or a valid FieldDescriptor::CppType.
  int type_;
};

// MapValueRef points to a map value.
class LIBPROTOBUF_EXPORT MapValueRef {
 public:
  MapValueRef() : data_(NULL), type_(0) {}

  void SetInt64Value(int64 value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT64,
               "MapValueRef::SetInt64Value");
    *reinterpret_cast<int64*>(data_) = value;
  }
  void SetUInt64Value(uint64 value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT64,
               "MapValueRef::SetUInt64Value");
    *reinterpret_cast<uint64*>(data_) = value;
  }
  void SetInt32Value(int32 value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT32,
               "MapValueRef::SetInt32Value");
    *reinterpret_cast<int32*>(data_) = value;
  }
  void SetUInt32Value(uint64 value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT32,
               "MapValueRef::SetUInt32Value");
    *reinterpret_cast<uint32*>(data_) = value;
  }
  void SetBoolValue(bool value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_BOOL,
               "MapValueRef::SetBoolValue");
    *reinterpret_cast<bool*>(data_) = value;
  }
  // TODO(jieluo) - Checks that enum is member.
  void SetEnumValue(int value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_ENUM,
               "MapValueRef::SetEnumValue");
    *reinterpret_cast<int*>(data_) = value;
  }
  void SetStringValue(const string& value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_STRING,
               "MapValueRef::SetStringValue");
    *reinterpret_cast<string*>(data_) = value;
  }
  void SetFloatValue(float value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_FLOAT,
               "MapValueRef::SetFloatValue");
    *reinterpret_cast<float*>(data_) = value;
  }
  void SetDoubleValue(double value) {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_DOUBLE,
               "MapValueRef::SetDoubleValue");
    *reinterpret_cast<double*>(data_) = value;
  }

  int64 GetInt64Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT64,
               "MapValueRef::GetInt64Value");
    return *reinterpret_cast<int64*>(data_);
  }
  uint64 GetUInt64Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT64,
               "MapValueRef::GetUInt64Value");
    return *reinterpret_cast<uint64*>(data_);
  }
  int32 GetInt32Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_INT32,
               "MapValueRef::GetInt32Value");
    return *reinterpret_cast<int32*>(data_);
  }
  uint32 GetUInt32Value() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_UINT32,
               "MapValueRef::GetUInt32Value");
    return *reinterpret_cast<uint32*>(data_);
  }
  bool GetBoolValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_BOOL,
               "MapValueRef::GetBoolValue");
    return *reinterpret_cast<bool*>(data_);
  }
  int GetEnumValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_ENUM,
               "MapValueRef::GetEnumValue");
    return *reinterpret_cast<int*>(data_);
  }
  const string& GetStringValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_STRING,
               "MapValueRef::GetStringValue");
    return *reinterpret_cast<string*>(data_);
  }
  float GetFloatValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_FLOAT,
               "MapValueRef::GetFloatValue");
    return *reinterpret_cast<float*>(data_);
  }
  double GetDoubleValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_DOUBLE,
               "MapValueRef::GetDoubleValue");
    return *reinterpret_cast<double*>(data_);
  }

  const Message& GetMessageValue() const {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_MESSAGE,
               "MapValueRef::GetMessageValue");
    return *reinterpret_cast<Message*>(data_);
  }

  Message* MutableMessageValue() {
    TYPE_CHECK(FieldDescriptor::CPPTYPE_MESSAGE,
               "MapValueRef::MutableMessageValue");
    return reinterpret_cast<Message*>(data_);
  }

 private:
  template <typename K, typename V,
            internal::WireFormatLite::FieldType key_wire_type,
            internal::WireFormatLite::FieldType value_wire_type,
            int default_enum_value>
  friend class internal::MapField;
  template <typename K, typename V>
  friend class internal::TypeDefinedMapFieldBase;
  friend class MapIterator;
  friend class internal::GeneratedMessageReflection;
  friend class internal::DynamicMapField;

  void SetType(FieldDescriptor::CppType type) {
    type_ = type;
  }

  FieldDescriptor::CppType type() const {
    if (type_ == 0 || data_ == NULL) {
      GOOGLE_LOG(FATAL)
          << "Protocol Buffer map usage error:\n"
          << "MapValueRef::type MapValueRef is not initialized.";
    }
    return (FieldDescriptor::CppType)type_;
  }
  void SetValue(const void* val) {
    data_ = const_cast<void*>(val);
  }
  void CopyFrom(const MapValueRef& other) {
    type_ = other.type_;
    data_ = other.data_;
  }
  // Only used in DynamicMapField
  void DeleteData() {
    switch (type_) {
#define HANDLE_TYPE(CPPTYPE, TYPE)                              \
      case google::protobuf::FieldDescriptor::CPPTYPE_##CPPTYPE: {        \
        delete reinterpret_cast<TYPE*>(data_);                  \
        break;                                                  \
      }
      HANDLE_TYPE(INT32, int32);
      HANDLE_TYPE(INT64, int64);
      HANDLE_TYPE(UINT32, uint32);
      HANDLE_TYPE(UINT64, uint64);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(STRING, string);
      HANDLE_TYPE(ENUM, int32);
      HANDLE_TYPE(MESSAGE, Message);
#undef HANDLE_TYPE
    }
  }
  // data_ point to a map value. MapValueRef does not
  // own this value.
  void* data_;
  // type_ is 0 or a valid FieldDescriptor::CppType.
  int type_;
};

#undef TYPE_CHECK

// This is the class for google::protobuf::Map's internal value_type. Instead of using
// std::pair as value_type, we use this class which provides us more control of
// its process of construction and destruction.
template <typename Key, typename T>
class MapPair {
 public:
  typedef const Key first_type;
  typedef T second_type;

  MapPair(const Key& other_first, const T& other_second)
      : first(other_first), second(other_second) {}
  explicit MapPair(const Key& other_first) : first(other_first), second() {}
  MapPair(const MapPair& other)
      : first(other.first), second(other.second) {}

  ~MapPair() {}

  // Implicitly convertible to std::pair of compatible types.
  template <typename T1, typename T2>
  operator std::pair<T1, T2>() const {
    return std::pair<T1, T2>(first, second);
  }

  const Key first;
  T second;

 private:
  friend class ::google::protobuf::Arena;
  friend class Map<Key, T>;
};

// google::protobuf::Map is an associative container type used to store protobuf map
// fields. Its interface is similar to std::unordered_map. Users should use this
// interface directly to visit or change map fields.
template <typename Key, typename T>
class Map {
 public:
  typedef Key key_type;
  typedef T mapped_type;
  typedef MapPair<Key, T> value_type;

  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;

  typedef size_t size_type;
  typedef hash<Key> hasher;
  typedef equal_to<Key> key_equal;

  Map()
      : arena_(NULL),
        allocator_(arena_),
        elements_(0, hasher(), key_equal(), allocator_),
        default_enum_value_(0) {}
  explicit Map(Arena* arena)
      : arena_(arena),
        allocator_(arena_),
        elements_(0, hasher(), key_equal(), allocator_),
        default_enum_value_(0) {
    arena_->OwnDestructor(&elements_);
  }

  Map(const Map& other)
      : arena_(NULL),
        allocator_(arena_),
        elements_(0, hasher(), key_equal(), allocator_),
        default_enum_value_(other.default_enum_value_) {
    insert(other.begin(), other.end());
  }
  template <class InputIt>
  explicit Map(const InputIt& first, const InputIt& last)
      : arena_(NULL),
        allocator_(arena_),
        elements_(0, hasher(), key_equal(), allocator_),
        default_enum_value_(0) {
    insert(first, last);
  }

  ~Map() { clear(); }

 private:
  // re-implement std::allocator to use arena allocator for memory allocation.
  // Used for google::protobuf::Map implementation. Users should not use this class
  // directly.
  template <typename U>
  class MapAllocator {
   public:
    typedef U value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    MapAllocator() : arena_(NULL) {}
    explicit MapAllocator(Arena* arena) : arena_(arena) {}
    template <typename X>
    MapAllocator(const MapAllocator<X>& allocator)
        : arena_(allocator.arena_) {}

    pointer allocate(size_type n, const_pointer hint = 0) {
      // If arena is not given, malloc needs to be called which doesn't
      // construct element object.
      if (arena_ == NULL) {
        return reinterpret_cast<pointer>(malloc(n * sizeof(value_type)));
      } else {
        return reinterpret_cast<pointer>(
            Arena::CreateArray<uint8>(arena_, n * sizeof(value_type)));
      }
    }

    void deallocate(pointer p, size_type n) {
      if (arena_ == NULL) {
        free(p);
      }
    }

#if __cplusplus >= 201103L && !defined(GOOGLE_PROTOBUF_OS_APPLE) && \
    !defined(GOOGLE_PROTOBUF_OS_NACL) && !defined(GOOGLE_PROTOBUF_OS_ANDROID)
    template<class NodeType, class... Args>
    void construct(NodeType* p, Args&&... args) {
      new (static_cast<void*>(p)) NodeType(std::forward<Args>(args)...);
    }

    template<class NodeType>
    void destroy(NodeType* p) {
      p->~NodeType();
    }
#else
    void construct(pointer p, const_reference t) { new (p) value_type(t); }

    void destroy(pointer p) { p->~value_type(); }
#endif

    template <typename X>
    struct rebind {
      typedef MapAllocator<X> other;
    };

    template <typename X>
    bool operator==(const MapAllocator<X>& other) const {
      return arena_ == other.arena_;
    }

    template <typename X>
    bool operator!=(const MapAllocator<X>& other) const {
      return arena_ != other.arena_;
    }

    // To support Visual Studio 2008
    size_type max_size() const {
      return std::numeric_limits<size_type>::max();
    }

   private:
    typedef void DestructorSkippable_;
    Arena* arena_;

    template <typename X>
    friend class MapAllocator;
  };

 public:
  typedef MapAllocator<std::pair<const Key, MapPair<Key, T>*> > Allocator;

  // Iterators
  class const_iterator
      : public std::iterator<std::forward_iterator_tag, value_type, ptrdiff_t,
                             const value_type*, const value_type&> {
    typedef typename hash_map<Key, value_type*, hash<Key>, equal_to<Key>,
                              Allocator>::const_iterator InnerIt;

   public:
    const_iterator() {}
    explicit const_iterator(const InnerIt& it) : it_(it) {}

    const_reference operator*() const { return *it_->second; }
    const_pointer operator->() const { return it_->second; }

    const_iterator& operator++() {
      ++it_;
      return *this;
    }
    const_iterator operator++(int) { return const_iterator(it_++); }

    friend bool operator==(const const_iterator& a, const const_iterator& b) {
      return a.it_ == b.it_;
    }
    friend bool operator!=(const const_iterator& a, const const_iterator& b) {
      return a.it_ != b.it_;
    }

   private:
    InnerIt it_;
  };

  class iterator : public std::iterator<std::forward_iterator_tag, value_type> {
    typedef typename hash_map<Key, value_type*, hasher, equal_to<Key>,
                              Allocator>::iterator InnerIt;

   public:
    iterator() {}
    explicit iterator(const InnerIt& it) : it_(it) {}

    reference operator*() const { return *it_->second; }
    pointer operator->() const { return it_->second; }

    iterator& operator++() {
      ++it_;
      return *this;
    }
    iterator operator++(int) { return iterator(it_++); }

    // Implicitly convertible to const_iterator.
    operator const_iterator() const { return const_iterator(it_); }

    friend bool operator==(const iterator& a, const iterator& b) {
      return a.it_ == b.it_;
    }
    friend bool operator!=(const iterator& a, const iterator& b) {
      return a.it_ != b.it_;
    }

   private:
    friend class Map;
    InnerIt it_;
  };

  iterator begin() { return iterator(elements_.begin()); }
  iterator end() { return iterator(elements_.end()); }
  const_iterator begin() const { return const_iterator(elements_.begin()); }
  const_iterator end() const { return const_iterator(elements_.end()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  // Capacity
  size_type size() const { return elements_.size(); }
  bool empty() const { return elements_.empty(); }

  // Element access
  T& operator[](const key_type& key) {
    value_type** value = &elements_[key];
    if (*value == NULL) {
      *value = CreateValueTypeInternal(key);
      internal::MapValueInitializer<google::protobuf::is_proto_enum<T>::value,
                                    T>::Initialize((*value)->second,
                                                   default_enum_value_);
    }
    return (*value)->second;
  }
  const T& at(const key_type& key) const {
    const_iterator it = find(key);
    GOOGLE_CHECK(it != end());
    return it->second;
  }
  T& at(const key_type& key) {
    iterator it = find(key);
    GOOGLE_CHECK(it != end());
    return it->second;
  }

  // Lookup
  size_type count(const key_type& key) const {
    return elements_.count(key);
  }
  const_iterator find(const key_type& key) const {
    return const_iterator(elements_.find(key));
  }
  iterator find(const key_type& key) {
    return iterator(elements_.find(key));
  }
  std::pair<const_iterator, const_iterator> equal_range(
      const key_type& key) const {
    const_iterator it = find(key);
    if (it == end()) {
      return std::pair<const_iterator, const_iterator>(it, it);
    } else {
      const_iterator begin = it++;
      return std::pair<const_iterator, const_iterator>(begin, it);
    }
  }
  std::pair<iterator, iterator> equal_range(const key_type& key) {
    iterator it = find(key);
    if (it == end()) {
      return std::pair<iterator, iterator>(it, it);
    } else {
      iterator begin = it++;
      return std::pair<iterator, iterator>(begin, it);
    }
  }

  // insert
  std::pair<iterator, bool> insert(const value_type& value) {
    iterator it = find(value.first);
    if (it != end()) {
      return std::pair<iterator, bool>(it, false);
    } else {
      return std::pair<iterator, bool>(
          iterator(elements_.insert(std::pair<Key, value_type*>(
              value.first, CreateValueTypeInternal(value))).first), true);
    }
  }
  template <class InputIt>
  void insert(InputIt first, InputIt last) {
    for (InputIt it = first; it != last; ++it) {
      iterator exist_it = find(it->first);
      if (exist_it == end()) {
        operator[](it->first) = it->second;
      }
    }
  }

  // Erase
  size_type erase(const key_type& key) {
    typename hash_map<Key, value_type*, hash<Key>, equal_to<Key>,
                      Allocator>::iterator it = elements_.find(key);
    if (it == elements_.end()) {
      return 0;
    } else {
      if (arena_ == NULL) delete it->second;
      elements_.erase(it);
      return 1;
    }
  }
  void erase(iterator pos) {
    if (arena_ == NULL) delete pos.it_->second;
    elements_.erase(pos.it_);
  }
  void erase(iterator first, iterator last) {
    for (iterator it = first; it != last;) {
      if (arena_ == NULL) delete it.it_->second;
      elements_.erase((it++).it_);
    }
  }
  void clear() {
    for (iterator it = begin(); it != end(); ++it) {
      if (arena_ == NULL) delete it.it_->second;
    }
    elements_.clear();
  }

  // Assign
  Map& operator=(const Map& other) {
    if (this != &other) {
      clear();
      insert(other.begin(), other.end());
    }
    return *this;
  }

 private:
  // Set default enum value only for proto2 map field whose value is enum type.
  void SetDefaultEnumValue(int default_enum_value) {
    default_enum_value_ = default_enum_value;
  }

  value_type* CreateValueTypeInternal(const Key& key) {
    if (arena_ == NULL) {
      return new value_type(key);
    } else {
      value_type* value = reinterpret_cast<value_type*>(
          Arena::CreateArray<uint8>(arena_, sizeof(value_type)));
      Arena::CreateInArenaStorage(const_cast<Key*>(&value->first), arena_);
      Arena::CreateInArenaStorage(&value->second, arena_);
      const_cast<Key&>(value->first) = key;
      return value;
    }
  }

  value_type* CreateValueTypeInternal(const value_type& value) {
    if (arena_ == NULL) {
      return new value_type(value);
    } else {
      value_type* p = reinterpret_cast<value_type*>(
          Arena::CreateArray<uint8>(arena_, sizeof(value_type)));
      Arena::CreateInArenaStorage(const_cast<Key*>(&p->first), arena_);
      Arena::CreateInArenaStorage(&p->second, arena_);
      const_cast<Key&>(p->first) = value.first;
      p->second = value.second;
      return p;
    }
  }

  Arena* arena_;
  Allocator allocator_;
  hash_map<Key, value_type*, hash<Key>, equal_to<Key>, Allocator> elements_;
  int default_enum_value_;

  friend class ::google::protobuf::Arena;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  template <typename K, typename V,
            internal::WireFormatLite::FieldType key_wire_type,
            internal::WireFormatLite::FieldType value_wire_type,
            int default_enum_value>
  friend class internal::MapFieldLite;
};

}  // namespace protobuf
}  // namespace google

GOOGLE_PROTOBUF_HASH_NAMESPACE_DECLARATION_START
template<>
struct hash<google::protobuf::MapKey> {
  size_t
  operator()(const google::protobuf::MapKey& map_key) const {
    switch (map_key.type()) {
      case google::protobuf::FieldDescriptor::CPPTYPE_STRING:
        return hash<string>()(map_key.GetStringValue());
      case google::protobuf::FieldDescriptor::CPPTYPE_INT64:
        return hash< ::google::protobuf::int64>()(map_key.GetInt64Value());
      case google::protobuf::FieldDescriptor::CPPTYPE_INT32:
        return hash< ::google::protobuf::int32>()(map_key.GetInt32Value());
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
        return hash< ::google::protobuf::uint64>()(map_key.GetUInt64Value());
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
        return hash< ::google::protobuf::uint32>()(map_key.GetUInt32Value());
      case google::protobuf::FieldDescriptor::CPPTYPE_BOOL:
        return hash<bool>()(map_key.GetBoolValue());
      case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      case google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Can't get here.";
        return 0;
    }
  }
  bool
  operator()(const google::protobuf::MapKey& map_key1,
             const google::protobuf::MapKey& map_key2) const {
    switch (map_key1.type()) {
#define COMPARE_CPPTYPE(CPPTYPE, CPPTYPE_METHOD)             \
      case google::protobuf::FieldDescriptor::CPPTYPE_##CPPTYPE: \
        return map_key1.Get##CPPTYPE_METHOD##Value() <           \
               map_key2.Get##CPPTYPE_METHOD##Value();
      COMPARE_CPPTYPE(STRING, String)
      COMPARE_CPPTYPE(INT64,  Int64)
      COMPARE_CPPTYPE(INT32,  Int32)
      COMPARE_CPPTYPE(UINT64, UInt64)
      COMPARE_CPPTYPE(UINT32, UInt32)
      COMPARE_CPPTYPE(BOOL,   Bool)
#undef COMPARE_CPPTYPE
      case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      case google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Can't get here.";
        return true;
    }
  }
};
GOOGLE_PROTOBUF_HASH_NAMESPACE_DECLARATION_END

#endif  // GOOGLE_PROTOBUF_MAP_H__
