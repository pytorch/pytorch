/*
 * strong_type C++14/17/20 strong typedef library
 *
 * Copyright (C) Björn Fahller
 *
 *  Use, modification and distribution is subject to the
 *  Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 *
 * Project home: https://github.com/rollbear/strong_type
 */

#ifndef ROLLBEAR_STRONG_TYPE_HPP_INCLUDED
#define ROLLBEAR_STRONG_TYPE_HPP_INCLUDED

#include <functional>
#include <istream>
#include <ostream>
#include <type_traits>
#include <utility>

#ifndef STRONG_HAS_STD_FORMAT
#define STRONG_HAS_STD_FORMAT 0
#endif

#ifndef STRONG_HAS_FMT_FORMAT
#define STRONG_HAS_FMT_FORMAT 0
#endif

#if STRONG_HAS_STD_FORMAT
#include <format>
#if !defined(__cpp_lib_format) || __cpp_lib_format < 201907
#undef STRONG_HAS_STD_FORMAT
#define STRONG_HAS_STD_FORMAT 0
#endif
#endif

#if STRONG_HAS_FMT_FORMAT
#include <fmt/format.h>
#endif

namespace strong
{

namespace impl
{
  template <typename T, typename ... V>
  using WhenConstructible = std::enable_if_t<std::is_constructible_v<T, V...>>;
}

template <typename M, typename T>
using modifier = typename M::template modifier<T>;

struct uninitialized_t {};
static constexpr uninitialized_t uninitialized{};

struct default_constructible
{
  template <typename T>
  class modifier
  {
  };
};

namespace impl {
  template <typename T>
  constexpr bool supports_default_construction(const ::strong::default_constructible::modifier<T>*)
  {
    return true;
  }
}

template <typename T, typename Tag, typename ... M>
class type : public modifier<M, type<T, Tag, M...>>...
{
public:
  template <typename TT = T, typename = std::enable_if_t<std::is_trivially_constructible<TT>{}>>
  explicit type(uninitialized_t)
    noexcept
  {
  }
  template <typename type_ = type,
            bool = impl::supports_default_construction(static_cast<type_*>(nullptr))>
  constexpr
  type()
    noexcept(noexcept(T{}))
  : val{}
  {
  }

  template <typename U,
    typename = impl::WhenConstructible<T, std::initializer_list<U>>>
  constexpr
  explicit
  type(
    std::initializer_list<U> us
  )
    noexcept(noexcept(T{us}))
  : val{us}
  {
  }
  template <typename ... U,
            typename = std::enable_if_t<std::is_constructible_v<T, U&&...> && (sizeof...(U) > 0)>>
  constexpr
  explicit
  type(
    U&& ... u)
  noexcept(std::is_nothrow_constructible_v<T, U...>)
  : val(std::forward<U>(u)...)
  {}

  friend constexpr void swap(type& a, type& b) noexcept(
                                                        std::is_nothrow_move_constructible_v<T> &&
                                                        std::is_nothrow_move_assignable_v<T>
                                                      )
  {
    using std::swap;
    swap(a.val, b.val);
  }

  [[nodiscard]]
  constexpr T& value_of() & noexcept { return val;}
  [[nodiscard]]
  constexpr const T& value_of() const & noexcept { return val;}
  [[nodiscard]]
  constexpr T&& value_of() && noexcept { return std::move(val);}

  [[nodiscard]]
  friend constexpr T& value_of(type& t) noexcept { return t.val;}
  [[nodiscard]]
  friend constexpr const T& value_of(const type& t) noexcept { return t.val;}
  [[nodiscard]]
  friend constexpr T&& value_of(type&& t) noexcept { return std::move(t).val;}
private:
  T val;
};

namespace impl {
  template <typename T, typename Tag, typename ... Ms>
  constexpr bool is_strong_type_func(const strong::type<T, Tag, Ms...>*) { return true;}
  constexpr bool is_strong_type_func(...) { return false;}
  template <typename T, typename Tag, typename ... Ms>
  constexpr T underlying_type(strong::type<T, Tag, Ms...>*);

}

template <typename T>
struct is_strong_type : std::integral_constant<bool, impl::is_strong_type_func(static_cast<T *>(nullptr))> {};

namespace impl {
  template <typename T>
  using WhenStrongType = std::enable_if_t<is_strong_type<std::decay_t<T>>::value>;
  template <typename T>
  using WhenNotStrongType = std::enable_if_t<!is_strong_type<std::decay_t<T>>::value>;
}

template <typename T, bool = is_strong_type<T>::value>
struct underlying_type
{
  using type = decltype(impl::underlying_type(static_cast<T*>(nullptr)));
};

template <typename T>
struct underlying_type<T, false>
{
  using type = T;
};

template <typename T>
using underlying_type_t = typename underlying_type<T>::type;


namespace impl {
  template<
    typename T,
    typename = impl::WhenNotStrongType<T>>
  constexpr
  T &&
  access(T &&t)
  noexcept {
    return std::forward<T>(t);
  }
  template <
    typename T,
    typename = impl::WhenStrongType<T>>
  [[nodiscard]]
  constexpr
  auto
  access(T&& t)
  noexcept
  -> decltype(value_of(std::forward<T>(t)))
  {
    return value_of(std::forward<T>(t));
  }

}
struct equality
{
  template <typename T>
  class modifier;
};


template <typename T, typename Tag, typename ... M>
class equality::modifier<::strong::type<T, Tag, M...>>
{
  using type = ::strong::type<T, Tag, M...>;
public:
  [[nodiscard]]
  friend
  constexpr
  auto
  operator==(
    const type& lh,
    const type& rh)
  noexcept(noexcept(std::declval<const T&>() == std::declval<const T&>()))
  -> decltype(std::declval<const T&>() == std::declval<const T&>())
  {
    return value_of(lh) == value_of(rh);
  }

  [[nodiscard]]
  friend
  constexpr
  auto
  operator!=(
    const type& lh,
    const type& rh)
  noexcept(noexcept(std::declval<const T&>() != std::declval<const T&>()))
  -> decltype(std::declval<const T&>() != std::declval<const T&>())
  {
    return value_of(lh) != value_of(rh);
  }
};

namespace impl
{
  template <typename T, typename Other>
  class typed_equality
  {
  private:
    using TT = underlying_type_t<T>;
    using OT = underlying_type_t<Other>;
  public:
    [[nodiscard]]
    friend
    constexpr
    auto operator==(const T& lh, const Other& rh)
    noexcept(noexcept(std::declval<const TT&>() == std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() == std::declval<const OT&>())
    {
      return value_of(lh) == impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator==(const Other& lh, const T& rh)
    noexcept(noexcept(std::declval<const OT&>() == std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() == std::declval<const TT&>())
    {
      return impl::access(lh) == value_of(rh) ;
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator!=(const T& lh, const Other rh)
    noexcept(noexcept(std::declval<const TT&>() != std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() != std::declval<const OT&>())
    {
      return value_of(lh) != impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator!=(const Other& lh, const T& rh)
    noexcept(noexcept(std::declval<const OT&>() != std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() != std::declval<const TT&>())
    {
      return impl::access(lh) != value_of(rh) ;
    }
  };
}
template <typename ... Ts>
struct equality_with
{
  template <typename T>
  class modifier : public impl::typed_equality<T, Ts>...
  {
  };
};

namespace impl
{
  template <typename T, typename Other>
  class typed_ordering
  {
  private:
    using TT = underlying_type_t<T>;
    using OT = underlying_type_t<Other>;
  public:
    [[nodiscard]]
    friend
    constexpr
    auto operator<(const T& lh, const Other& rh)
    noexcept(noexcept(std::declval<const TT&>() < std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() < std::declval<const OT&>())
    {
      return value_of(lh) < impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator<(const Other& lh, const T& rh)
    noexcept(noexcept(std::declval<const OT&>() < std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() < std::declval<const TT&>())
    {
      return impl::access(lh) < value_of(rh) ;
    }

    [[nodiscard]]
    friend
    constexpr
    auto operator<=(const T& lh, const Other& rh)
    noexcept(noexcept(std::declval<const TT&>() <= std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() <= std::declval<const OT&>())
    {
      return value_of(lh) <= impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator<=(const Other& lh, const T& rh)
    noexcept(noexcept(std::declval<const OT&>() <= std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() <= std::declval<const TT&>())
    {
      return impl::access(lh) <= value_of(rh) ;
    }

    [[nodiscard]]
    friend
    constexpr
    auto operator>(const T& lh, const Other& rh)
    noexcept(noexcept(std::declval<const TT&>() > std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() > std::declval<const OT&>())
    {
      return value_of(lh) > impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator>(const Other& lh, const T& rh)
    noexcept(noexcept(std::declval<const OT&>() > std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() > std::declval<const TT&>())
    {
      return impl::access(lh) > value_of(rh) ;
    }

    [[nodiscard]]
    friend
    constexpr
    auto operator>=(const T& lh, const Other& rh)
    noexcept(noexcept(std::declval<const TT&>() >= std::declval<const OT&>()))
    -> decltype(std::declval<const TT&>() >= std::declval<const OT&>())
    {
      return value_of(lh) >= impl::access(rh);
    }
    [[nodiscard]]
    friend
    constexpr
    auto operator>=(const Other& lh, const T& rh)
    noexcept(noexcept(std::declval<const OT&>() >= std::declval<const TT&>()))
    -> decltype(std::declval<const OT&>() >= std::declval<const TT&>())
    {
      return impl::access(lh) >= value_of(rh) ;
    }
  };
}

template <typename ... Ts>
struct ordered_with
{
  template <typename T>
  class modifier : public impl::typed_ordering<T, Ts>...
  {
  };
};

namespace impl
{
  template <typename T>
  struct require_copy_constructible
  {
    static constexpr bool value = std::is_copy_constructible<underlying_type_t<T>>::value;
    static_assert(value, "underlying type must be copy constructible");
  };
  template <typename T>
  struct require_move_constructible
  {
    static constexpr bool value = std::is_move_constructible<underlying_type_t<T>>::value;
    static_assert(value, "underlying type must be move constructible");
  };
  template <typename T>
  struct require_copy_assignable
  {
    static constexpr bool value = std::is_copy_assignable<underlying_type_t<T>>::value;
    static_assert(value, "underlying type must be copy assignable");
  };
  template <typename T>
  struct require_move_assignable
  {
    static constexpr bool value = std::is_move_assignable<underlying_type_t<T>>::value;
    static_assert(value, "underlying type must be move assignable");
  };

  template <bool> struct valid_type;
  template <>
  struct valid_type<true> {};

  template <typename T>
  struct require_semiregular
    : valid_type<require_copy_constructible<T>::value &&
                 require_move_constructible<T>::value &&
                 require_copy_assignable<T>::value &&
                 require_move_assignable<T>::value>
  {
  };

}
struct semiregular
{
  template <typename>
  class modifier;
};

template <typename T, typename Tag, typename ... M>
class semiregular::modifier<::strong::type<T, Tag, M...>>
  : public default_constructible::modifier<T>
  , private impl::require_semiregular<T>
{
};

struct regular
{
  template <typename T>
  class modifier
    : public semiregular::modifier<T>
    , public equality::modifier<T>
  {
  };
};

struct unique
{
  template <typename T>
  class modifier
    : private impl::valid_type<
      impl::require_move_constructible<T>::value &&
      impl::require_move_assignable<T>::value
    >
  {
  public:
    constexpr modifier() = default;
    modifier(const modifier&) = delete;
    constexpr modifier(modifier&&) = default;
    modifier& operator=(const modifier&) = delete;
    constexpr modifier& operator=(modifier&&) = default;
  };
};
struct ordered
{
  template <typename T>
  class modifier;
};


template <typename T, typename Tag, typename ... M>
class ordered::modifier<::strong::type<T, Tag, M...>>
{
  using type = ::strong::type<T, Tag, M...>;
public:
  [[nodiscard]]
  friend
  constexpr
  auto
  operator<(
    const type& lh,
    const type& rh)
  noexcept(noexcept(std::declval<const T&>() < std::declval<const T&>()))
  -> decltype(std::declval<const T&>() < std::declval<const T&>())
  {
    return value_of(lh) < value_of(rh);
  }

  [[nodiscard]]
  friend
  constexpr
  auto
  operator<=(
    const type& lh,
    const type& rh)
  noexcept(noexcept(std::declval<const T&>() <= std::declval<const T&>()))
  -> decltype(std::declval<const T&>() <= std::declval<const T&>())
  {
    return value_of(lh) <= value_of(rh);
  }

  [[nodiscard]]
  friend
  constexpr
  auto
  operator>(
    const type& lh,
    const type& rh)
  noexcept(noexcept(std::declval<const T&>() > std::declval<const T&>()))
  -> decltype(std::declval<const T&>() > std::declval<const T&>())
  {
    return value_of(lh) > value_of(rh);
  }

  [[nodiscard]]
  friend
  constexpr

  auto
  operator>=(
    const type& lh,
    const type& rh)
  noexcept(noexcept(std::declval<const T&>() >= std::declval<const T&>()))
  -> decltype(std::declval<const T&>() >= std::declval<const T&>())
  {
    return value_of(lh) >= value_of(rh);
  }
};

struct ostreamable
{
  template <typename T>
  class modifier
  {
  public:
    friend
    std::ostream&
    operator<<(
      std::ostream &os,
      const T &t)
    {
      return os << value_of(t);
    }
  };
};

struct istreamable
{
  template <typename T>
  class modifier
  {
  public:
    friend
    std::istream&
    operator>>(
      std::istream &is,
      T &t)
    {
      return is >> value_of(t);
    }
  };
};

struct iostreamable
{
  template <typename T>
  class modifier
    : public ostreamable::modifier<T>
    , public istreamable::modifier<T>
  {
  };
};

struct incrementable
{
  template <typename T>
  class modifier
  {
  public:
    friend
    constexpr
    T&
    operator++(T& t)
    noexcept(noexcept(++std::declval<T&>().value_of()))
    {
      ++value_of(t);
      return t;
    }

    friend
    constexpr
    T
    operator++(T& t, int)
    {
      auto copy = t;
      ++t;
      return copy;
    }
  };
};

struct decrementable
{
  template <typename T>
  class modifier
  {
  public:
    friend
    constexpr
    T&
    operator--(T& t)
    noexcept(noexcept(--std::declval<T&>().value_of()))
    {
      --value_of(t);
      return t;
    }

    friend
    constexpr
    T
    operator--(T& t, int)
    {
      auto copy = t;
      --t;
      return copy;
    }
  };
};

struct bicrementable
{
  template <typename T>
  class modifier
    : public incrementable::modifier<T>
    , public decrementable::modifier<T>
  {
  };
};

struct boolean
{
  template <typename T>
  class modifier
  {
  public:
    explicit constexpr operator bool() const
    noexcept(noexcept(static_cast<bool>(value_of(std::declval<const T&>()))))
    {
      const auto& self = static_cast<const T&>(*this);
      return static_cast<bool>(value_of(self));
    }
  };
};

struct hashable
{
  template <typename T>
  class modifier{};
};

struct difference
{
  template <typename T>
  class modifier;
};

template <typename T, typename Tag, typename ... M>
class difference::modifier<::strong::type<T, Tag, M...>>
: public ordered::modifier<::strong::type<T, Tag, M...>>
, public equality::modifier<::strong::type<T, Tag, M...>>
{
  using type = ::strong::type<T, Tag, M...>;
public:
  friend
  constexpr
  type& operator+=(type& lh, const type& rh)
  noexcept(noexcept(value_of(lh) += value_of(rh)))
  {
    value_of(lh) += value_of(rh);
    return lh;
  }

  friend
  constexpr
  type& operator-=(type& lh, const type& rh)
    noexcept(noexcept(value_of(lh) -= value_of(rh)))
  {
    value_of(lh) -= value_of(rh);
    return lh;
  }

  friend
  constexpr
  type& operator*=(type& lh, const T& rh)
  noexcept(noexcept(value_of(lh) *= rh))
  {
    value_of(lh) *= rh;
    return lh;
  }

  friend
  constexpr
  type& operator/=(type& lh, const T& rh)
    noexcept(noexcept(value_of(lh) /= rh))
  {
    value_of(lh) /= rh;
    return lh;
  }

  template <typename TT = T, typename = decltype(std::declval<TT&>()%= std::declval<const TT&>())>
  friend
  constexpr
  type& operator%=(type& lh, const T& rh)
    noexcept(noexcept(value_of(lh) %= rh))
  {
    value_of(lh)%= rh;
    return lh;
  }

  friend
  constexpr
  type operator+(type lh, const type& rh)
  {
    lh += rh;
    return lh;
  }

  friend
  constexpr
  type operator-(type lh, const type& rh)
  {
    lh -= rh;
    return lh;
  }

  friend
  constexpr
  type operator*(type lh, const T& rh)
  {
    lh *= rh;
    return lh;
  }

  friend
  constexpr
  type operator*(const T& lh, type rh)
  {
    rh *= lh;
    return rh;
  }

  friend
  constexpr
  type operator/(type lh, const T& rh)
  {
    lh /= rh;
    return lh;
  }

  friend
  constexpr
  T operator/(const type& lh, const type& rh)
  {
    return value_of(lh) / value_of(rh);
  }

  template <typename TT = T, typename = decltype(std::declval<TT&>() %= std::declval<const TT&>())>
  friend
  constexpr
  type operator%(type lh, const T& rh)
    noexcept(noexcept(lh%= rh))
  {
      lh %= rh;
      return lh;
  }

  template <typename TT = T, typename = decltype(std::declval<TT>() % std::declval<TT>())>
  friend
  constexpr
  T operator%(type lh, type rh)
    noexcept(noexcept(value_of(lh) % value_of(rh)))
  {
      return value_of(lh) % value_of(rh);
  }
};

template <typename D = void>
struct affine_point
{
  template <typename T>
  class modifier;
};

namespace impl
{
  template <typename ...>
  using void_t = void;

  template <typename T, typename = void>
  struct subtractable : std::false_type {};

  template <typename T>
  struct subtractable<T, void_t<decltype(std::declval<const T&>() - std::declval<const T&>())>>
  : std::true_type {};
}


template <typename D>
template <typename T, typename Tag, typename ... M>
class affine_point<D>::modifier<::strong::type<T, Tag, M...>>
{
  using type = ::strong::type<T, Tag, M...>;
  static_assert(impl::subtractable<T>::value, "it must be possible to subtract instances of your underlying type");
  using base_diff_type = decltype(std::declval<const T&>() - std::declval<const T&>());
public:
  using difference = std::conditional_t<std::is_same<D, void>{}, strong::type<base_diff_type, Tag, strong::difference>, D>;
  static_assert(std::is_constructible_v<difference, base_diff_type>, "");
  [[nodiscard]]
  friend
  constexpr
  difference
  operator-(
    const type& lh,
    const type& rh)
  {
    return difference(value_of(lh) - value_of(rh));
  }

  friend
  constexpr
  type&
  operator+=(
    type& lh,
    const difference& d)
  noexcept(noexcept(value_of(lh) += impl::access(d)))
  {
    value_of(lh) += impl::access(d);
    return lh;
  }

  friend
  constexpr
  type&
  operator-=(
    type& lh,
    const difference& d)
  noexcept(noexcept(value_of(lh) -= impl::access(d)))
  {
    value_of(lh) -= impl::access(d);
    return lh;
  }

  [[nodiscard]]
  friend
  constexpr
  type
  operator+(
    type lh,
    const difference& d)
  {
    return lh += d;
  }

  [[nodiscard]]
  friend
  constexpr
  type
  operator+(
    const difference& d,
    type rh)
  {
    return rh+= d;
  }

  [[nodiscard]]
  friend
  constexpr
  type
  operator-(
    type lh,
    const difference& d)
  {
    return lh -= d;
  }
};


struct pointer
{
  template <typename T>
  class modifier;
};

template <typename T, typename Tag, typename ... M>
class pointer::modifier<::strong::type<T, Tag, M...>>
{
  using type = strong::type<T, Tag, M...>;
public:
  template <typename TT = T>
  [[nodiscard]]
  friend
  constexpr
  auto
  operator==(
    const type& t,
    std::nullptr_t)
  noexcept(noexcept(std::declval<const TT&>() == nullptr))
  -> decltype(std::declval<const TT&>() == nullptr)
  {
    return value_of(t) == nullptr;
  }

  template <typename TT = T>
  [[nodiscard]]
  friend
  constexpr
  auto
  operator==(
    std::nullptr_t,
    const type& t)
  noexcept(noexcept(nullptr == std::declval<const TT&>()))
  -> decltype(nullptr == std::declval<const TT&>())
  {
    return value_of(t) == nullptr;
  }

  template <typename TT = T>
  [[nodiscard]]
  friend
  constexpr
  auto
  operator!=(
    const type& t,
    std::nullptr_t)
  noexcept(noexcept(std::declval<const TT&>() != nullptr))
  -> decltype(std::declval<const TT&>() != nullptr)
  {
    return value_of(t) != nullptr;
  }

  template <typename TT = T>
  [[nodiscard]]
  friend
  constexpr
  auto
  operator!=(
    std::nullptr_t,
    const type& t)
  noexcept(noexcept(nullptr != std::declval<const TT&>()))
  -> decltype(nullptr != std::declval<const TT&>())
  {
    return value_of(t) != nullptr;
  }

  [[nodiscard]]
  constexpr
  decltype(*std::declval<const T&>())
  operator*()
  const
  {
    auto& self = static_cast<const type&>(*this);
    return *value_of(self);
  }

  [[nodiscard]]
  constexpr
  decltype(&(*std::declval<const T&>())) operator->() const { return &operator*();}
};

struct arithmetic
{
  template <typename T>
  class modifier
  {
  public:
    [[nodiscard]]
    friend
    constexpr
    T
    operator-(
      const T &lh)
    {
      return T{-value_of(lh)};
    }

    friend
    constexpr
    T&
    operator+=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) += value_of(rh)))
    {
      value_of(lh) += value_of(rh);
      return lh;
    }

    friend
    constexpr
    T&
    operator-=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) -= value_of(rh)))
    {
      value_of(lh) -= value_of(rh);
      return lh;
    }

    friend
    constexpr
    T&
    operator*=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) *= value_of(rh)))
    {
      value_of(lh) *= value_of(rh);
      return lh;
    }

    friend
    constexpr
    T&
    operator/=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) /= value_of(rh)))
    {
      value_of(lh) /= value_of(rh);
      return lh;
    }

    template <typename TT = T, typename = decltype(value_of(std::declval<TT>()) % value_of(std::declval<TT>()))>
    friend
    constexpr
    T&
    operator%=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) %= value_of(rh)))
    {
      value_of(lh) %= value_of(rh);
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator+(
      T lh,
      const T &rh)
    {
      lh += rh;
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator-(
      T lh,
      const T &rh)
    {
      lh -= rh;
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator*(
      T lh,
      const T &rh)
    {
      lh *= rh;
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator/(
      T lh,
      const T &rh)
    {
      lh /= rh;
      return lh;
    }

    template <typename TT = T, typename = decltype(value_of(std::declval<TT>()) % value_of(std::declval<TT>()))>
    [[nodiscard]]
    friend
    constexpr
    T
    operator%(
      T lh,
      const T &rh)
    {
      lh %= rh;
      return lh;
    }

  };
};


struct bitarithmetic
{
  template <typename T>
  class modifier
  {
  public:
    friend
    constexpr
    T&
    operator&=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) &= value_of(rh)))
    {
      value_of(lh) &= value_of(rh);
      return lh;
    }

    friend
    constexpr
    T&
    operator|=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) |= value_of(rh)))
    {
      value_of(lh) |= value_of(rh);
      return lh;
    }

    friend
    constexpr
    T&
    operator^=(
      T &lh,
      const T &rh)
    noexcept(noexcept(value_of(lh) ^= value_of(rh)))
    {
      value_of(lh) ^= value_of(rh);
      return lh;
    }

    template <typename C>
    friend
    constexpr
    T&
    operator<<=(
      T &lh,
      C c)
    noexcept(noexcept(value_of(lh) <<= c))
    {
      value_of(lh) <<= c;
      return lh;
    }

    template <typename C>
    friend
    constexpr
    T&
    operator>>=(
      T &lh,
      C c)
    noexcept(noexcept(value_of(lh) >>= c))
    {
      value_of(lh) >>= c;
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator~(
      const T &lh)
    {
      auto v = value_of(lh);
      v = ~v;
      return T(v);
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator&(
      T lh,
      const T &rh)
    {
      lh &= rh;
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator|(
      T lh,
      const T &rh)
    {
      lh |= rh;
      return lh;
    }

    [[nodiscard]]
    friend
    constexpr
    T
    operator^(
      T lh,
      const T &rh)
    {
      lh ^= rh;
      return lh;
    }

    template <typename C>
    [[nodiscard]]
    friend
    constexpr
    T
    operator<<(
      T lh,
      C c)
    {
      lh <<= c;
      return lh;
    }

    template <typename C>
    [[nodiscard]]
    friend
    constexpr
    T
    operator>>(
      T lh,
      C c)
    {
      lh >>= c;
      return lh;
    }
  };
};
template <typename I = void>
struct indexed
{
  template <typename T>
  class modifier;
};

template <>
struct indexed<void> {
  template<typename>
  class modifier;

  template <typename T, typename Tag, typename ... Ms>
  class modifier<type<T, Tag, Ms...>> {
    using ref = T&;
    using cref = const T&;
    using rref = T&&;
    using type = strong::type<T, Tag, Ms...>;
  public:
    template<typename I>
    [[nodiscard]]
    auto
    operator[](
      const I &i)
    const &
    noexcept(noexcept(std::declval<cref>()[impl::access(i)]))
    -> decltype(std::declval<cref>()[impl::access(i)]) {
      auto& self = static_cast<const type&>(*this);
      return value_of(self)[impl::access(i)];
    }

    template<typename I>
    [[nodiscard]]
    auto
    operator[](
      const I &i)
    &
    noexcept(noexcept(std::declval<ref>()[impl::access(i)]))
    -> decltype(std::declval<ref>()[impl::access(i)]) {
      auto& self = static_cast<type&>(*this);
      return value_of(self)[impl::access(i)];
    }

    template<typename I>
    [[nodiscard]]
    auto
    operator[](
      const I &i)
    &&
    noexcept(noexcept(std::declval<rref>()[impl::access(i)]))
    -> decltype(std::declval<rref>()[impl::access(i)]) {
      auto& self = static_cast<type&>(*this);
      return value_of(std::move(self))[impl::access(i)];
    }

    template<typename I, typename C = cref>
    [[nodiscard]]
    auto
    at(
      const I &i)
    const &
    -> decltype(std::declval<C>().at(impl::access(i))) {
      auto& self = static_cast<const type&>(*this);
      return value_of(self).at(impl::access(i));
    }

    template<typename I, typename R = ref>
    [[nodiscard]]
    auto
    at(
      const I &i)
    &
    -> decltype(std::declval<R>().at(impl::access(i))) {
      auto& self = static_cast<type&>(*this);
      return value_of(self).at(impl::access(i));
    }

    template<typename I, typename R = rref>
    [[nodiscard]]
    auto
    at(
      const I &i)
    &&
    -> decltype(std::declval<R>().at(impl::access(i))) {
      auto& self = static_cast<type&>(*this);
      return value_of(std::move(self)).at(impl::access(i));
    }
  };
};

template <typename I>
template <typename T, typename Tag, typename ... M>
class indexed<I>::modifier<type<T, Tag, M...>>
{
  using type = ::strong::type<T, Tag, M...>;
public:
  [[nodiscard]]
  auto
  operator[](
    const I& i)
  const &
  noexcept(noexcept(std::declval<const T&>()[impl::access(i)]))
  -> decltype(std::declval<const T&>()[impl::access(i)])
  {
    auto& self = static_cast<const type&>(*this);
    return value_of(self)[impl::access(i)];
  }

  [[nodiscard]]
  auto
  operator[](
    const I& i)
  &
  noexcept(noexcept(std::declval<T&>()[impl::access(i)]))
  -> decltype(std::declval<T&>()[impl::access(i)])
  {
    auto& self = static_cast<type&>(*this);
    return value_of(self)[impl::access(i)];
  }

  [[nodiscard]]
  auto
  operator[](
    const I& i)
  &&
  noexcept(noexcept(std::declval<T&&>()[impl::access(i)]))
  -> decltype(std::declval<T&&>()[impl::access(i)])
  {
    auto& self = static_cast<type&>(*this);
    return value_of(std::move(self))[impl::access(i)];
  }

  template <typename TT = T>
  [[nodiscard]]
  auto
  at(
    const I& i)
  const &
  -> decltype(std::declval<const TT&>().at(impl::access(i)))
  {
    auto& self = static_cast<const type&>(*this);
    return value_of(self).at(impl::access(i));
  }

  template <typename TT = T>
  [[nodiscard]]
  auto
  at(
    const I& i)
  &
  -> decltype(std::declval<TT&>().at(impl::access(i)))
  {
    auto& self = static_cast<type&>(*this);
    return value_of(self).at(impl::access(i));
  }

  template <typename TT = T>
  [[nodiscard]]
  auto
  at(
    const I& i)
  &&
  -> decltype(std::declval<TT&&>().at(impl::access(i)))
  {
    auto& self = static_cast<type&>(*this);
    return value_of(std::move(self)).at(impl::access(i));
  }
};

class iterator
{
public:
  template <typename I, typename category = typename std::iterator_traits<underlying_type_t<I>>::iterator_category>
  class modifier
    : public pointer::modifier<I>
    , public equality::modifier<I>
    , public incrementable::modifier<I>
  {
  public:
    using difference_type = typename std::iterator_traits<underlying_type_t<I>>::difference_type;
    using value_type = typename std::iterator_traits<underlying_type_t<I>>::value_type;
    using pointer = typename std::iterator_traits<underlying_type_t<I>>::value_type;
    using reference = typename std::iterator_traits<underlying_type_t<I>>::reference;
    using iterator_category = typename std::iterator_traits<underlying_type_t<I>>::iterator_category;
  };

  template <typename I>
  class modifier<I, std::bidirectional_iterator_tag>
    : public modifier<I, std::forward_iterator_tag>
      , public decrementable::modifier<I>
  {
  };
  template <typename I>
  class modifier<I, std::random_access_iterator_tag>
    : public modifier<I, std::bidirectional_iterator_tag>
      , public affine_point<typename std::iterator_traits<underlying_type_t<I>>::difference_type>::template modifier<I>
      , public indexed<>::modifier<I>
      , public ordered::modifier<I>
  {
  };
};

class range
{
public:
  template <typename R>
  class modifier;
};

template <typename T, typename Tag, typename ... M>
class range::modifier<type<T, Tag, M...>>
{
  using type = ::strong::type<T, Tag, M...>;
  using r_iterator = decltype(std::declval<T&>().begin());
  using r_const_iterator = decltype(std::declval<const T&>().begin());
public:
  using iterator = ::strong::type<r_iterator, Tag, strong::iterator>;
  using const_iterator = ::strong::type<r_const_iterator, Tag, strong::iterator>;

  iterator
  begin()
  noexcept(noexcept(std::declval<T&>().begin()))
  {
    auto& self = static_cast<type&>(*this);
    return iterator{value_of(self).begin()};
  }

  iterator
  end()
  noexcept(noexcept(std::declval<T&>().end()))
  {
    auto& self = static_cast<type&>(*this);
    return iterator{value_of(self).end()};
  }

  const_iterator
  cbegin()
    const
  noexcept(noexcept(std::declval<const T&>().begin()))
  {
    auto& self = static_cast<const type&>(*this);
    return const_iterator{value_of(self).begin()};
  }

  const_iterator
  cend()
    const
  noexcept(noexcept(std::declval<const T&>().end()))
  {
    auto& self = static_cast<const type&>(*this);
    return const_iterator{value_of(self).end()};
  }

  const_iterator
  begin()
  const
  noexcept(noexcept(std::declval<const T&>().begin()))
  {
    auto& self = static_cast<const type&>(*this);
    return const_iterator{value_of(self).begin()};
  }

  const_iterator
  end()
  const
  noexcept(noexcept(std::declval<const T&>().end()))
  {
    auto& self = static_cast<const type&>(*this);
    return const_iterator{value_of(self).end()};
  }
};

namespace impl {

  template<typename T, typename D>
  struct converter
  {
    constexpr explicit operator D() const
    noexcept(noexcept(static_cast<D>(std::declval<const underlying_type_t<T>&>())))
    {
      auto& self = static_cast<const T&>(*this);
      return static_cast<D>(value_of(self));
    }
  };
  template<typename T, typename D>
  struct implicit_converter
  {
    constexpr operator D() const
    noexcept(noexcept(static_cast<D>(std::declval<const underlying_type_t<T>&>())))
    {
      auto& self = static_cast<const T&>(*this);
      return static_cast<D>(value_of(self));
    }
  };
}
template <typename ... Ts>
struct convertible_to
{
  template <typename T>
  struct modifier : impl::converter<T, Ts>...
  {
  };
};

template <typename ... Ts>
struct implicitly_convertible_to
{
  template <typename T>
  struct modifier : impl::implicit_converter<T, Ts>...
  {
  };

};

struct formattable
{
    template <typename T>
    class modifier{};
};

}

namespace std {
template <typename T, typename Tag, typename ... M>
struct hash<::strong::type<T, Tag, M...>>
  : std::conditional_t<
    std::is_base_of<
      ::strong::hashable::modifier<
        ::strong::type<T, Tag, M...>
      >,
      ::strong::type<T, Tag, M...>
    >::value,
    hash<T>,
    std::false_type>
{
  using type = ::strong::type<T, Tag, M...>;
  decltype(auto)
  operator()(
    const ::strong::hashable::modifier<type>& t)
  const
  noexcept(noexcept(std::declval<hash<T>>()(value_of(std::declval<const type&>()))))
  {
    auto& tt = static_cast<const type&>(t);
    return hash<T>::operator()(value_of(tt));
  }
};
template <typename T, typename Tag, typename ... M>
struct is_arithmetic<::strong::type<T, Tag, M...>>
  : is_base_of<::strong::arithmetic::modifier<::strong::type<T, Tag, M...>>,
               ::strong::type<T, Tag, M...>>
{
};

#if STRONG_HAS_STD_FORMAT
template<typename T, typename Tag, typename... M, typename Char>
struct formatter<::strong::type<T, Tag, M...>, Char,
                 std::enable_if_t<
                     std::is_base_of<
                         ::strong::formattable::modifier<
                             ::strong::type<T, Tag, M...>
                             >,
                         ::strong::type<T, Tag, M...>
                         >::value
                     >>
    : formatter<T>
{
  using type = ::strong::type<T, Tag, M...>;
  template<typename FormatContext>
  constexpr
  decltype(auto)
  format(const ::strong::formattable::modifier<type>& t, FormatContext& fc)
      noexcept(noexcept(std::declval<formatter<T, Char>>().format(value_of(std::declval<const type&>()), fc)))
  {
    const auto& tt = static_cast<const type&>(t);
    return formatter<T, Char>::format(value_of(tt), fc);
  }
};
#endif

}

#if STRONG_HAS_FMT_FORMAT
namespace fmt
{
template<typename T, typename Tag, typename... M, typename Char>
struct formatter<::strong::type<T, Tag, M...>, Char,
                 std::enable_if_t<
                   std::is_base_of<
                     ::strong::formattable::modifier<
                       ::strong::type<T, Tag, M...>
                     >,
                     ::strong::type<T, Tag, M...>
                   >::value
                 >>
  : formatter<T>
{
  using type = ::strong::type<T, Tag, M...>;
  template<typename FormatContext>
  constexpr
  decltype(auto)
  format(const ::strong::formattable::modifier<type>& t, FormatContext& fc)
      noexcept(noexcept(std::declval<formatter<T, Char>>().format(value_of(std::declval<const type&>()), fc)))
  {
    const auto& tt = static_cast<const type&>(t);
    return formatter<T, Char>::format(value_of(tt), fc);
  }
};
}
#endif
#endif //ROLLBEAR_STRONG_TYPE_HPP_INCLUDED
