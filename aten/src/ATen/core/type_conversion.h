#pragma once

#include <ATen/core/jit_type.h>
#include <unordered_map>
#include <atomic>

template <class ValueType>
class TypeMap {
  // Internally, we'll use a hash table to store mapping from type
  // IDs to the values.
  typedef std::unordered_map<int, ValueType> InternalMap;
public:
  typedef typename InternalMap::iterator iterator;
  typedef typename InternalMap::const_iterator const_iterator;
  typedef typename InternalMap::value_type value_type;

  const_iterator begin() const { return m_map.begin(); }
  const_iterator end() const { return m_map.end();  }
  iterator begin() { return m_map.begin();  }
  iterator end() { return m_map.end(); }

  // Finds the value associated with the type "Key" in the type map.
  template <class Key>
  iterator find() { return m_map.find(getTypeId<Key>());  }

  // Same as above, const version
  template <class Key>
  const_iterator find() const { return m_map.find(getTypeId<Key>()); }

  // Associates a value with the type "Key"
  template <class Key>
  void put(ValueType &&value) {
    m_map[getTypeId<Key>()] = std::forward<ValueType>(value);
  }
  int size() {
      return m_map.size();
  }

private:
  template <class Key>
  inline static int getTypeId() {
    static const int id = LastTypeId++;
    return id;
  }

  static std::atomic_int LastTypeId;
  InternalMap m_map;
};

template <class ValueType>
std::atomic_int TypeMap<ValueType>::LastTypeId(0);

static TypeMap<c10::ClassTypePtr> tmap;

namespace c10 {
namespace detail {
template <typename T>
struct getTypePtr_ final {
  static TypePtr call() {
    auto res = tmap.find<T>();
    if (res == tmap.end()) {
        throw c10::Error("Trying to convert a class that's not registered.", "");
    }
    return std::dynamic_pointer_cast<Type>(res->second);
  }
//   static_assert(
//       guts::false_t<T>::value,
//       "Type could not be converted to any of the known types.");
};

template <>
struct getTypePtr_<at::Tensor> final {
  static TypePtr call() {
    return TensorType::get();
  }
};
template <>
struct getTypePtr_<double> final {
  static TypePtr call() {
    return FloatType::get();
  }
};
template <>
struct getTypePtr_<int64_t> final {
  static TypePtr call() {
    return IntType::get();
  }
};
template <>
struct getTypePtr_<bool> final {
  static TypePtr call() {
    return BoolType::get();
  }
};
template <>
struct getTypePtr_<at::Scalar> final {
  static TypePtr call() {
    return NumberType::get();
  }
};
template <>
struct getTypePtr_<std::string> final {
  static TypePtr call() {
    return StringType::get();
  }
};
template <class T>
struct getTypePtr_<std::vector<T>> final {
  static TypePtr call() {
    static auto type = ListType::create(getTypePtr_<T>::call());
    return type;
  }
};
template <class T>
struct getTypePtr_<c10::ArrayRef<T>> final {
  static TypePtr call() {
    static auto type = ListType::create(getTypePtr_<T>::call());
    return type;
  }
};
template <class T>
struct getTypePtr_<c10::ListPtr<T>> final {
  static TypePtr call() {
    static auto type = ListType::create(getTypePtr_<T>::call());
    return type;
  }
};
template <class K, class V>
struct getTypePtr_<std::unordered_map<K, V>> final {
  static TypePtr call() {
    static auto type =
        DictType::create(getTypePtr_<K>::call(), getTypePtr_<V>::call());
    return type;
  }
};
template <class K, class V>
struct getTypePtr_<c10::DictPtr<K, V>> final {
  static TypePtr call() {
    static auto type =
        DictType::create(getTypePtr_<K>::call(), getTypePtr_<V>::call());
    return type;
  }
};
template <class T>
struct getTypePtr_<at::optional<T>> final {
  static TypePtr call() {
    static auto type = OptionalType::create(getTypePtr_<T>::call());
    return type;
  }
};
} // namespace detail
template <class T>
inline TypePtr getTypePtr() {
  // TODO: static_assert that a templated function exists, and throw a friendy
  // error message if not
  return detail::getTypePtr_<T>::call();
}
namespace ivalue {
  #define CREATE_FACTORY_INIT(type) inline IValue from(type v) { return IValue(v); }
  #define CONVERSION_TYPES(_) \
  _(at::Tensor)\
  _(intrusive_ptr<caffe2::Blob>)\
  _(intrusive_ptr<Capsule>)\
  _(ivalue::TuplePtr)\
  _(double)\
  _(c10::intrusive_ptr<ivalue::Future>)\
  _(int64_t)\
  _(int32_t)\
  _(bool)\
  _(c10::ListPtr<int64_t>)\
  _(c10::ArrayRef<int64_t>)\
  _(std::vector<int64_t>)\
  _(c10::intrusive_ptr<ivalue::ConstantString>)\
  _(std::basic_string<char>)\
  _(c10::ListPtr<double>)\
  _(std::vector<double>)\
  _(c10::ListPtr<bool>)\
  _(std::vector<bool>)\
  _(c10::ListPtr<at::Tensor>)\
  _(std::vector<at::Tensor>)\
  _(std::vector<IValue>)\
  _(c10::ListPtr<IValue>)\
  _(c10::intrusive_ptr<ivalue::Object>)\
  _(at::Scalar)\
  _(c10::Device)\
  _(c10::nullopt_t)\

  CONVERSION_TYPES(CREATE_FACTORY_INIT)

  template<class T>
  IValue from(c10::ListPtr<T> v) { return IValue(v); }
  template<class T>
  IValue from(std::vector<T> v) { return IValue(v); }
  inline IValue from(c10::DictPtr<IValue, IValue> v) { return IValue(v); }
  template<class Key, class Value>
  IValue from(c10::DictPtr<Key, Value> v) { return IValue(v); }
  template<class Key, class Value>
  IValue from(std::unordered_map<Key, Value> v) { return IValue(v); }
  template<class T>
  IValue from(c10::optional<T> v) { return IValue(v); }
  template <typename T>
  IValue from(T x) {
    auto res = tmap.find<T>();
    if (res == tmap.end()) {
      throw c10::Error("Trying to return a class that we don't support and isn't a registered custom class.", "");
    }
    auto retObject = ivalue::Object(res->second, 1);
    auto capsule = Capsule();
    capsule.ptr = (void*)x;
    retObject.setAttr("capsule", IValue(c10::make_intrusive<Capsule>(std::move(capsule))));
    auto resIVal = IValue(c10::make_intrusive<ivalue::Object>(std::move(retObject)));
    return resIVal;
      // static_assert(guts::false_t<T>::value, "You tried to register a kernel with an unsupported argument type.");

  }
}
} // namespace c10