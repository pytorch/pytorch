#pragma once

/**
 * Simple registry implementation that uses static variables to
 * register object creators during program initialization time.
 */

// NB: This Registry works poorly when you have other namespaces.
// Make all macro invocations from inside the at namespace.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>

#include <ATen/Error.h>
#include <ATen/ATenGeneral.h>

namespace at {

template <typename KeyType>
inline void PrintOffendingKey(const KeyType& /*key*/) {
  printf("[key type printing not supported]\n");
}

template <>
inline void PrintOffendingKey(const std::string& key) {
  printf("Offending key: %s.\n", key.c_str());
}

/**
 * @brief A template class that allows one to register classes by keys.
 *
 * The keys are usually a std::string specifying the name, but can be anything that
 * can be used in a std::map.
 *
 * You should most likely not use the Registry class explicitly, but use the
 * helper macros below to declare specific registries as well as registering
 * objects.
 */
template <class SrcType, class ObjectPtrType, class... Args>
class AT_API Registry {
 public:
  typedef std::function<ObjectPtrType(Args...)> Creator;

  Registry() : registry_() {}

  void Register(const SrcType& key, Creator creator) {
    // The if statement below is essentially the same as the following line:
    // CHECK_EQ(registry_.count(key), 0) << "Key " << key
    //                                   << " registered twice.";
    // However, CHECK_EQ depends on google logging, and since registration is
    // carried out at static initialization time, we do not want to have an
    // explicit dependency on glog's initialization function.
    std::lock_guard<std::mutex> lock(register_mutex_);
    if (registry_.count(key) != 0) {
      printf("Key already registered.\n");
      PrintOffendingKey(key);
      std::exit(1);
    }
    registry_[key] = creator;
  }

  void Register(const SrcType& key, Creator creator, const std::string& help_msg) {
    Register(key, creator);
    help_message_[key] = help_msg;
  }

  inline bool Has(const SrcType& key) { return (registry_.count(key) != 0); }

  ObjectPtrType Create(const SrcType& key, Args... args) {
    if (registry_.count(key) == 0) {
      // Returns nullptr if the key is not registered.
      return nullptr;
    }
    return registry_[key](args...);
  }

  /**
   * Returns the keys currently registered as a std::vector.
   */
  std::vector<SrcType> Keys() {
    std::vector<SrcType> keys;
    for (const auto& it : registry_) {
      keys.push_back(it.first);
    }
    return keys;
  }

  const std::unordered_map<SrcType, std::string>& HelpMessage() const {
    return help_message_;
  }

  const char* HelpMessage(const SrcType& key) const {
    auto it = help_message_.find(key);
    if (it == help_message_.end()) {
      return nullptr;
    }
    return it->second.c_str();
  }

 private:
  std::unordered_map<SrcType, Creator> registry_;
  std::unordered_map<SrcType, std::string> help_message_;
  std::mutex register_mutex_;

  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;
};

template <class SrcType, class ObjectPtrType, class... Args>
class AT_API Registerer {
 public:
  Registerer(
      const SrcType& key,
      Registry<SrcType, ObjectPtrType, Args...>* registry,
      typename Registry<SrcType, ObjectPtrType, Args...>::Creator creator,
      const std::string& help_msg = "") {
    registry->Register(key, creator, help_msg);
  }

  template <class DerivedType>
  static ObjectPtrType DefaultCreator(Args... args) {
    // TODO(jiayq): old versions of NVCC does not handle make_unique well
    // so we are forced to use a unique_ptr constructor here. Check if it is
    // fine to use make_unique in the future.
    // return make_unique<DerivedType>(args...);
    return ObjectPtrType(new DerivedType(args...));
  }
};

/**
 * AT_ANONYMOUS_VARIABLE(str) introduces an identifier starting with
 * str and ending with a number that varies with the line.
 * Pretty much a copy from 'folly/Preprocessor.h'
 */
#define AT_CONCATENATE_IMPL(s1, s2) s1##s2
#define AT_CONCATENATE(s1, s2) AT_CONCATENATE_IMPL(s1, s2)
#ifdef __COUNTER__
#define AT_ANONYMOUS_VARIABLE(str) AT_CONCATENATE(str, __COUNTER__)
#else
#define AT_ANONYMOUS_VARIABLE(str) AT_CONCATENATE(str, __LINE__)
#endif

/**
 * AT_DECLARE_TYPED_REGISTRY is a macro that expands to a function
 * declaration, as well as creating a convenient typename for its corresponding
 * registerer.
 */
#define AT_DECLARE_TYPED_REGISTRY(                                    \
    RegistryName, SrcType, ObjectType, PtrType, ...)                     \
  AT_API Registry<SrcType, PtrType<ObjectType>, __VA_ARGS__>* RegistryName(); \
  typedef Registerer<SrcType, PtrType<ObjectType>, __VA_ARGS__>        \
      Registerer##RegistryName; \
  extern template class Registerer<SrcType, PtrType<ObjectType>, __VA_ARGS__>;

#define AT_DEFINE_TYPED_REGISTRY(                                         \
    RegistryName, SrcType, ObjectType, PtrType, ...)                         \
  Registry<SrcType, PtrType<ObjectType>, __VA_ARGS__>* RegistryName() {    \
    static Registry<SrcType, PtrType<ObjectType>, __VA_ARGS__>* registry = \
        new Registry<SrcType, PtrType<ObjectType>, __VA_ARGS__>();         \
    return registry;                                                         \
  } \
  template class Registerer<SrcType, PtrType<ObjectType>, __VA_ARGS__>;

// Note(Yangqing): The __VA_ARGS__ below allows one to specify a templated
// creator with comma in its templated arguments.
#define AT_REGISTER_TYPED_CREATOR(RegistryName, key, ...)                  \
  namespace {                                                                 \
  Registerer##RegistryName AT_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, RegistryName(), __VA_ARGS__);                                      \
  }

#define AT_REGISTER_TYPED_CLASS(RegistryName, key, ...)                    \
  namespace {                                                                 \
  Registerer##RegistryName AT_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                                    \
      RegistryName(),                                                         \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                  \
      ::at::demangle_type<__VA_ARGS__>());                                           \
  }

// AT_DECLARE_REGISTRY and AT_DEFINE_REGISTRY are hard-wired to use std::string
// as the key
// type, because that is the most commonly used cases.
#define AT_DECLARE_REGISTRY(RegistryName, ObjectType, ...) \
  AT_DECLARE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, __VA_ARGS__)

#define AT_DEFINE_REGISTRY(RegistryName, ObjectType, ...) \
  AT_DEFINE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, __VA_ARGS__)

#define AT_DECLARE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  AT_DECLARE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, __VA_ARGS__)

#define AT_DEFINE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  AT_DEFINE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, __VA_ARGS__)

// AT_REGISTER_CREATOR and AT_REGISTER_CLASS are hard-wired to use std::string
// as the key
// type, because that is the most commonly used cases.
#define AT_REGISTER_CREATOR(RegistryName, key, ...) \
  AT_REGISTER_TYPED_CREATOR(RegistryName, #key, __VA_ARGS__)

#define AT_REGISTER_CLASS(RegistryName, key, ...) \
  AT_REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

}  // namespace at
