#ifndef C10_UTIL_REGISTRY_H_
#define C10_UTIL_REGISTRY_H_

/**
 * Simple registry implementation that uses static variables to
 * register object creators during program initialization time.
 */

// NB: This Registry works poorly when you have other namespaces.
// Make all macro invocations from inside the at namespace.

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Type.h>

namespace c10 {

template <typename KeyType>
inline std::string KeyStrRepr(const KeyType& /*key*/) {
  return "[key type printing not supported]";
}

template <>
inline std::string KeyStrRepr(const std::string& key) {
  return key;
}

enum RegistryPriority {
  REGISTRY_FALLBACK = 1,
  REGISTRY_DEFAULT = 2,
  REGISTRY_PREFERRED = 3,
};

/**
 * @brief A template class that allows one to register classes by keys.
 *
 * The keys are usually a std::string specifying the name, but can be anything
 * that can be used in a std::map.
 *
 * You should most likely not use the Registry class explicitly, but use the
 * helper macros below to declare specific registries as well as registering
 * objects.
 */
template <class SrcType, class ObjectPtrType, class... Args>
class Registry {
 public:
  typedef std::function<ObjectPtrType(Args...)> Creator;

  Registry(bool warning = true) : registry_(), priority_(), warning_(warning) {}
  ~Registry() = default;

  void Register(
      const SrcType& key,
      Creator creator,
      const RegistryPriority priority = REGISTRY_DEFAULT) {
    std::lock_guard<std::mutex> lock(register_mutex_);
    // The if statement below is essentially the same as the following line:
    // TORCH_CHECK_EQ(registry_.count(key), 0) << "Key " << key
    //                                   << " registered twice.";
    // However, TORCH_CHECK_EQ depends on google logging, and since registration
    // is carried out at static initialization time, we do not want to have an
    // explicit dependency on glog's initialization function.
    if (registry_.count(key) != 0) {
      auto cur_priority = priority_[key];
      if (priority > cur_priority) {
#ifdef DEBUG
        std::string warn_msg =
            "Overwriting already registered item for key " + KeyStrRepr(key);
        fprintf(stderr, "%s\n", warn_msg.c_str());
#endif
        registry_[key] = creator;
        priority_[key] = priority;
      } else if (priority == cur_priority) {
        std::string err_msg =
            "Key already registered with the same priority: " + KeyStrRepr(key);
        fprintf(stderr, "%s\n", err_msg.c_str());
        if (terminate_) {
          std::exit(1);
        } else {
          throw std::runtime_error(err_msg);
        }
      } else if (warning_) {
        std::string warn_msg =
            "Higher priority item already registered, skipping registration of " +
            KeyStrRepr(key);
        fprintf(stderr, "%s\n", warn_msg.c_str());
      }
    } else {
      registry_[key] = creator;
      priority_[key] = priority;
    }
  }

  void Register(
      const SrcType& key,
      Creator creator,
      const std::string& help_msg,
      const RegistryPriority priority = REGISTRY_DEFAULT) {
    Register(key, creator, priority);
    help_message_[key] = help_msg;
  }

  inline bool Has(const SrcType& key) {
    return (registry_.count(key) != 0);
  }

  ObjectPtrType Create(const SrcType& key, Args... args) {
    auto it = registry_.find(key);
    if (it == registry_.end()) {
      // Returns nullptr if the key is not registered.
      return nullptr;
    }
    return it->second(args...);
  }

  /**
   * Returns the keys currently registered as a std::vector.
   */
  std::vector<SrcType> Keys() const {
    std::vector<SrcType> keys;
    keys.reserve(registry_.size());
    for (const auto& it : registry_) {
      keys.push_back(it.first);
    }
    return keys;
  }

  inline const std::unordered_map<SrcType, std::string>& HelpMessage() const {
    return help_message_;
  }

  const char* HelpMessage(const SrcType& key) const {
    auto it = help_message_.find(key);
    if (it == help_message_.end()) {
      return nullptr;
    }
    return it->second.c_str();
  }

  // Used for testing, if terminate is unset, Registry throws instead of
  // calling std::exit
  void SetTerminate(bool terminate) {
    terminate_ = terminate;
  }

  C10_DISABLE_COPY_AND_ASSIGN(Registry);
  Registry(Registry&&) = delete;
  Registry& operator=(Registry&&) = delete;

 private:
  std::unordered_map<SrcType, Creator> registry_;
  std::unordered_map<SrcType, RegistryPriority> priority_;
  bool terminate_{true};
  const bool warning_;
  std::unordered_map<SrcType, std::string> help_message_;
  std::mutex register_mutex_;
};

template <class SrcType, class ObjectPtrType, class... Args>
class Registerer {
 public:
  explicit Registerer(
      const SrcType& key,
      Registry<SrcType, ObjectPtrType, Args...>* registry,
      typename Registry<SrcType, ObjectPtrType, Args...>::Creator creator,
      const std::string& help_msg = "") {
    registry->Register(key, creator, help_msg);
  }

  explicit Registerer(
      const SrcType& key,
      const RegistryPriority priority,
      Registry<SrcType, ObjectPtrType, Args...>* registry,
      typename Registry<SrcType, ObjectPtrType, Args...>::Creator creator,
      const std::string& help_msg = "") {
    registry->Register(key, creator, help_msg, priority);
  }

  template <class DerivedType>
  static ObjectPtrType DefaultCreator(Args... args) {
    return ObjectPtrType(new DerivedType(args...));
  }
};

/**
 * C10_DECLARE_TYPED_REGISTRY is a macro that expands to a function
 * declaration, as well as creating a convenient typename for its corresponding
 * registerer.
 */
// Note on C10_IMPORT and C10_EXPORT below: we need to explicitly mark DECLARE
// as import and DEFINE as export, because these registry macros will be used
// in downstream shared libraries as well, and one cannot use *_API - the API
// macro will be defined on a per-shared-library basis. Semantically, when one
// declares a typed registry it is always going to be IMPORT, and when one
// defines a registry (which should happen ONLY ONCE and ONLY IN SOURCE FILE),
// the instantiation unit is always going to be exported.
//
// The only unique condition is when in the same file one does DECLARE and
// DEFINE - in Windows compilers, this generates a warning that dllimport and
// dllexport are mixed, but the warning is fine and linker will be properly
// exporting the symbol. Same thing happens in the gflags flag declaration and
// definition caes.
#define C10_DECLARE_TYPED_REGISTRY(                                      \
    RegistryName, SrcType, ObjectType, PtrType, ...)                     \
  C10_API ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*  \
  RegistryName();                                                        \
  typedef ::c10::Registerer<SrcType, PtrType<ObjectType>, ##__VA_ARGS__> \
      Registerer##RegistryName

#define TORCH_DECLARE_TYPED_REGISTRY(                                     \
    RegistryName, SrcType, ObjectType, PtrType, ...)                      \
  TORCH_API ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>* \
  RegistryName();                                                         \
  typedef ::c10::Registerer<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>  \
      Registerer##RegistryName

#define C10_DEFINE_TYPED_REGISTRY(                                         \
    RegistryName, SrcType, ObjectType, PtrType, ...)                       \
  C10_EXPORT ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>* \
  RegistryName() {                                                         \
    static ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*   \
        registry = new ::c10::                                             \
            Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>();       \
    return registry;                                                       \
  }

#define C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(                            \
    RegistryName, SrcType, ObjectType, PtrType, ...)                          \
  C10_EXPORT ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*    \
  RegistryName() {                                                            \
    static ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*      \
        registry =                                                            \
            new ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>( \
                false);                                                       \
    return registry;                                                          \
  }

// Note(Yangqing): The __VA_ARGS__ below allows one to specify a templated
// creator with comma in its templated arguments.
#define C10_REGISTER_TYPED_CREATOR(RegistryName, key, ...)                  \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, RegistryName(), ##__VA_ARGS__);

#define C10_REGISTER_TYPED_CREATOR_WITH_PRIORITY(                           \
    RegistryName, key, priority, ...)                                       \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, priority, RegistryName(), ##__VA_ARGS__);

#define C10_REGISTER_TYPED_CLASS(RegistryName, key, ...)                    \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                                  \
      RegistryName(),                                                       \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                \
      ::c10::demangle_type<__VA_ARGS__>());

#define C10_REGISTER_TYPED_CLASS_WITH_PRIORITY(                             \
    RegistryName, key, priority, ...)                                       \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                                  \
      priority,                                                             \
      RegistryName(),                                                       \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                \
      ::c10::demangle_type<__VA_ARGS__>());

// C10_DECLARE_REGISTRY and C10_DEFINE_REGISTRY are hard-wired to use
// std::string as the key type, because that is the most commonly used cases.
#define C10_DECLARE_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DECLARE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define TORCH_DECLARE_REGISTRY(RegistryName, ObjectType, ...) \
  TORCH_DECLARE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define C10_DEFINE_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define C10_DEFINE_REGISTRY_WITHOUT_WARNING(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define C10_DECLARE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DECLARE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

#define TORCH_DECLARE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  TORCH_DECLARE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

#define C10_DEFINE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

#define C10_DEFINE_SHARED_REGISTRY_WITHOUT_WARNING( \
    RegistryName, ObjectType, ...)                  \
  C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(        \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

// C10_REGISTER_CREATOR and C10_REGISTER_CLASS are hard-wired to use std::string
// as the key
// type, because that is the most commonly used cases.
#define C10_REGISTER_CREATOR(RegistryName, key, ...) \
  C10_REGISTER_TYPED_CREATOR(RegistryName, #key, __VA_ARGS__)

#define C10_REGISTER_CREATOR_WITH_PRIORITY(RegistryName, key, priority, ...) \
  C10_REGISTER_TYPED_CREATOR_WITH_PRIORITY(                                  \
      RegistryName, #key, priority, __VA_ARGS__)

#define C10_REGISTER_CLASS(RegistryName, key, ...) \
  C10_REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

#define C10_REGISTER_CLASS_WITH_PRIORITY(RegistryName, key, priority, ...) \
  C10_REGISTER_TYPED_CLASS_WITH_PRIORITY(                                  \
      RegistryName, #key, priority, __VA_ARGS__)

} // namespace c10

#endif // C10_UTIL_REGISTRY_H_
