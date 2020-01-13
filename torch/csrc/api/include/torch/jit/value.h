#pragma once
// #include <torch/csrc/jit/script/module.h>
// #include <torch/jit/type.h>

// namespace torch {

// // All the methods that go on a Value get defined here. This makes it easy for
// // accessor objects, used to represent the LHS of assignments to also have these
// // methods through inheritance. This design is adapted from pybind11
// using jit::TypePtr;

// template <typename T>
// struct _fake_type {};


// template <typename Derived>
// struct ValueAPI;

// template <typename Derived, typename Elem>
// std::vector<Elem> generic_to(
//     const ValueAPI<Derived>* ivalue,
//     _fake_type<std::vector<Elem>>);

// template <typename Derived, typename K, typename V>
// std::unordered_map<K, V> generic_to(
//     const ValueAPI<Derived>* ivalue,
//     _fake_type<std::unordered_map<K, V>>);


// template <typename Derived>
// struct ValueAPI {
//   // example to, but in the real thing
//   // we should wrap by hand, so we have control over what is publically exposed
//   template <typename T>
//   T to() const {
//     // static_assert(sizeof(T) == 0, "Only specializations of .to<T> can be used");
//     generic_to(this, _fake_type<T>{});
//   };

//   template <typename T>
//   T is() const {
//     static_assert(sizeof(T) == 0, "Only specializations of .is<T> can be used");
//   };

//   template<>
//   int64_t to<int64_t>() const {
//     return derived().toInt();
//   }

//   template<>
//   int64_t is<int64_t>() const {
//     return derived().isInt();
//   }

//   template <typename T>
//   struct _fake_type {};

//   Type type() const {
//     return derived().type();
//   }

//   bool isinstance(const Type& t) {
//     return type().isSubtypeOf(t);
//   }

//  private:
//   const jit::IValue& derived() const {
//     return static_cast<const Derived&>(*this).ivalue();
//   }
// };

// template <typename Derived, typename Elem>
// std::vector<Elem> generic_to(
//     const ValueAPI<Derived>* ivalue,
//     _fake_type<std::vector<Elem>>) {
//   return fmap(ivalue->derived()->toGenericListRef(), [](c10::IValue item_ivalue) {
//     return item_ivalue.to<Elem>();
//   });
// }


// struct Value : ValueAPI<Value> {
//   // ctors and to forward to Ivalue in prototype, but should be hand exposed in
//   // real thing
//   template <typename T>
//   Value(T&& value) : value_(std::forward<T>(value)) {}
//   Value(Value& value);
//   Value(Value&& value);
//   Value(const Value& value);

//   const jit::IValue& ivalue() const {
//     return value_;
//   }

//  protected:
//   jit::IValue value_;
// };


// } // namespace torch
