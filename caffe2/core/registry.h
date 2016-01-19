#ifndef CAFFE2_CORE_REGISTRY_H_
#define CAFFE2_CORE_REGISTRY_H_

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>

#include "caffe2/core/common.h"
#include "caffe2/core/typeid.h"

namespace caffe2 {

// Registry is a class that allows one to register classes by a specific
// key, usually a string specifying the name. For each key type and object type,
// there should be only one single registry responsible for it.

template <class SrcType, class ObjectType, class... Args>
class Registry {
 public:
  typedef std::function<ObjectType*(Args ...)> Creator;

  Registry() : registry_() {}

  void Register(const SrcType& key, Creator creator) {
    // The if statement below is essentially the same as the following line:
    // CAFFE_CHECK_EQ(registry_.count(key), 0) << "Key " << key
    //                                   << " registered twice.";
    // However, CAFFE_CHECK_EQ depends on google logging, and since registration is
    // carried out at static initialization time, we do not want to have an
    // explicit dependency on glog's initialization function.
    std::lock_guard<std::mutex> lock(register_mutex_);
    if (registry_.count(key) != 0) {
      std::cerr << "Key " << key << " already registered." << std::endl;
      std::exit(1);
    }
    //std::cout << "Registering " << key << " for "
    //          << typeid(ObjectType).name() << " creator.";
    registry_[key] = creator;
  }

  void Register(const SrcType& key, Creator creator, const string& help_msg) {
    Register(key, creator);
    help_message_[key] = help_msg;
  }

  inline bool Has(const SrcType& key) { return (registry_.count(key) != 0); }

  ObjectType* Create(const SrcType& key, Args ... args) {
    if (registry_.count(key) == 0) {
      // std::cerr << "Key " << key << " not found." << std::endl;
      // std::cerr << "Available keys:" << std::endl;
      // TODO: do we always want to print out the registered names? Sounds a bit
      // too verbose.
      //TEST_PrintRegisteredNames();
      // std::cerr << "Returning null pointer.";
      return nullptr;
    }
    return registry_[key](args...);
  }

  /**
   * Returns the keys currently registered as a vector.
   */
  vector<SrcType> Keys() {
    vector<SrcType> keys;
    for (const auto& it : registry_) {
      keys.push_back(it.first);
    }
    return keys;
  }

  const CaffeMap<SrcType, string>& HelpMessage() { return help_message_; }

 private:
  CaffeMap<SrcType, Creator> registry_;
  CaffeMap<SrcType, string> help_message_;
  std::mutex register_mutex_;

  DISABLE_COPY_AND_ASSIGN(Registry);
};

template <class SrcType, class ObjectType, class... Args>
class Registerer {
 public:
  Registerer(const SrcType& key,
             Registry<SrcType, ObjectType, Args...>* registry,
             typename Registry<SrcType, ObjectType, Args...>::Creator creator,
             const string& help_msg="") {
    registry->Register(key, creator, help_msg);
  }

  template <class DerivedType>
  static ObjectType* DefaultCreator(Args ... args) {
    return new DerivedType(args...);
  }
};

#define DECLARE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...)         \
  Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName();                \
  typedef Registerer<SrcType, ObjectType, ##__VA_ARGS__>                       \
      Registerer##RegistryName;

#define DEFINE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...)          \
  Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName() {               \
    static Registry<SrcType, ObjectType, ##__VA_ARGS__>* registry =            \
        new Registry<SrcType, ObjectType, ##__VA_ARGS__>();                    \
    return registry;                                                           \
  }

// For the typed creators, since we cannot affix the key as a simple string to
// the registerer object, we ask the user to provide an affix that we can
// affix to the registerer name to ensure uniqueness.
// Note(Yangqing): The __VA_ARGS__ below allows one to specify a templated
// creator with comma in its templated arguments.
#define REGISTER_TYPED_CREATOR(RegistryName, affix, key, ...)                  \
  Registerer##RegistryName g_##RegistryName##_##affix(                         \
      key, RegistryName(), __VA_ARGS__)
#define REGISTER_TYPED_CLASS(RegistryName, affix, key, ...)                    \
  Registerer##RegistryName g_##RegistryName##_##affix(                         \
      key, RegistryName(),                                                     \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>)

// DECLARE_REGISTRY and DEFINE_REGISTRY are hard-wired to use string as the key
// type, because that is the most commonly used cases.
#define DECLARE_REGISTRY(RegistryName, ObjectType, ...)                        \
  DECLARE_TYPED_REGISTRY(RegistryName, std::string,                            \
                         ObjectType, ##__VA_ARGS__)

#define DEFINE_REGISTRY(RegistryName, ObjectType, ...)                         \
  DEFINE_TYPED_REGISTRY(RegistryName, std::string,                             \
                        ObjectType, ##__VA_ARGS__)

// REGISTER_CREATOR and REGISTER_CLASS are hard-wired to use string as the key
// type, because that is the most commonly used cases.
#define REGISTER_CREATOR(RegistryName, key, ...)                               \
  REGISTER_TYPED_CREATOR(RegistryName, key, #key, __VA_ARGS__)

#define REGISTER_CLASS(RegistryName, key, ...)                                 \
  REGISTER_TYPED_CLASS(RegistryName, key, #key, __VA_ARGS__)

}  // namespace caffe2
#endif  // CAFFE2_CORE_REGISTRY_H_
