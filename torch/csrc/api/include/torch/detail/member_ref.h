#include <torch/parameter_list.h>

#include <torch/csrc/autograd/variable.h>

#include <functional>
#include <type_traits>

namespace torch { namespace detail {

/// A class that stores const-correct getters to a member variable, accessible
/// through a pointer to an object.
template <typename T>
class MemberRef {
 public:
  // TODO: Replace with (std/boost/our)::any.
  using This = void*;
  using ConstThis = const void*;
  using Getter = std::function<T&(This)>;
  using ConstGetter = std::function<const T&(ConstThis)>;

  template <typename Class>
  /* implicit */ MemberRef(T Class::*member)
      : getter_([member](This self) -> T& {
          return static_cast<Class*>(self)->*member;
        }),
        const_getter_([member](ConstThis self) -> const T& {
          return static_cast<const Class*>(self)->*member;
        }) {}

  template <typename Class>
  MemberRef(std::vector<T> Class::*member, size_t index)
      : getter_([member, index](This self) -> T& {
          return (static_cast<Class*>(self)->*member)[index];
        }),
        const_getter_([member, index](ConstThis self) -> const T& {
          return (static_cast<const Class*>(self)->*member)[index];
        }) {}

  // TODO: Replace ParameterList with OrderedDict when it's there
  template <
      typename Class,
      typename U = T,
      typename = typename std::enable_if<
          std::is_same<U, autograd::Variable>::value>::type>
  /* implicit */ MemberRef(ParameterList Class::*member, size_t index)
      : getter_([member, index](This self) -> T& {
          return (static_cast<Class*>(self)->*member)[index];
        }),
        const_getter_([member, index](ConstThis self) -> const T& {
          return (static_cast<const Class*>(self)->*member)[index];
        }) {}

  template <typename Class>
  T& operator()(Class* object) {
    return getter_(object);
  }

  template <typename Class>
  const T& operator()(const Class* object) const {
    return const_getter_(object);
  }

 private:
  Getter getter_;
  ConstGetter const_getter_;
};

}} // namespace torch::detail
