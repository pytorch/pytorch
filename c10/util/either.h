// Originally taken from
// https://github.com/cryfs/cryfs/blob/14ad22570ddacef22d5ff139cdff68a54fc8234d/src/cpp-utils/either.h

#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>
#include <c10/util/Optional.h>

namespace c10 {
/**
 * either<A, B> is a tagged union that holds either an object of type A
 * or an object of type B.
 */
template <class Left, class Right>
class either final {
 public:
  template <
      class Head,
      class... Tail,
      std::enable_if_t<
          std::is_constructible<Left, Head, Tail...>::value &&
          !std::is_constructible<Right, Head, Tail...>::value>* = nullptr>
  either(Head&& construct_left_head_arg, Tail&&... construct_left_tail_args)
      : _side(Side::left) {
    _construct_left(
        std::forward<Head>(construct_left_head_arg),
        std::forward<Tail>(construct_left_tail_args)...);
  }

  template <
      class Head,
      class... Tail,
      std::enable_if_t<
          !std::is_constructible<Left, Head, Tail...>::value &&
          std::is_constructible<Right, Head, Tail...>::value>* = nullptr>
  either(Head&& construct_right_head_arg, Tail&&... construct_right_tail_args)
      : _side(Side::right) {
    _construct_right(
        std::forward<Head>(construct_right_head_arg),
        std::forward<Tail>(construct_right_tail_args)...);
  }

  either(const either<Left, Right>& rhs) : _side(rhs._side) {
    if (_side == Side::left) {
      _construct_left(
          rhs._left); // NOLINT(cppcoreguidelines-pro-type-union-access)
    } else {
      _construct_right(
          rhs._right); // NOLINT(cppcoreguidelines-pro-type-union-access)
    }
  }

  either(either<Left, Right>&& rhs) noexcept : _side(rhs._side) {
    if (_side == Side::left) {
      _construct_left(std::move(
          rhs._left)); // NOLINT(cppcoreguidelines-pro-type-union-access)
    } else {
      _construct_right(std::move(
          rhs._right)); // NOLINT(cppcoreguidelines-pro-type-union-access)
    }
  }

  ~either() {
    _destruct();
  }

  either<Left, Right>& operator=(const either<Left, Right>& rhs) {
    _destruct();
    _side = rhs._side;
    if (_side == Side::left) {
      _construct_left(
          rhs._left); // NOLINT(cppcoreguidelines-pro-type-union-access)
    } else {
      _construct_right(
          rhs._right); // NOLINT(cppcoreguidelines-pro-type-union-access)
    }
    return *this;
  }

  either<Left, Right>& operator=(either<Left, Right>&& rhs) noexcept {
    _destruct();
    _side = rhs._side;
    if (_side == Side::left) {
      _construct_left(std::move(
          rhs._left)); // NOLINT(cppcoreguidelines-pro-type-union-access)
    } else {
      _construct_right(std::move(
          rhs._right)); // NOLINT(cppcoreguidelines-pro-type-union-access)
    }
    return *this;
  }

  bool is_left() const noexcept {
    return _side == Side::left;
  }

  bool is_right() const noexcept {
    return _side == Side::right;
  }

  const Left& left() const& {
    if (C10_UNLIKELY(!is_left())) {
      throw std::logic_error(
          "Tried to get left side of an either which is right.");
    }
    return _left; // NOLINT(cppcoreguidelines-pro-type-union-access)
  }
  Left& left() & {
    return const_cast<Left&>(
        const_cast<const either<Left, Right>*>(this)->left());
  }
  Left&& left() && {
    return std::move(left());
  }

  const Right& right() const& {
    if (C10_UNLIKELY(!is_right())) {
      throw std::logic_error(
          "Tried to get right side of an either which is left.");
    }
    return _right; // NOLINT(cppcoreguidelines-pro-type-union-access)
  }
  Right& right() & {
    return const_cast<Right&>(
        const_cast<const either<Left, Right>*>(this)->right());
  }
  Right&& right() && {
    return std::move(right());
  }

  template <class Result, class LeftFoldFunc, class RightFoldFunc>
  Result fold(LeftFoldFunc&& leftFoldFunc, RightFoldFunc&& rightFoldFunc)
      const {
    if (Side::left == _side) {
      return std::forward<LeftFoldFunc>(leftFoldFunc)(_left);
    } else {
      return std::forward<RightFoldFunc>(rightFoldFunc)(_right);
    }
  }

 private:
  union {
    Left _left;
    Right _right;
  };
  enum class Side : uint8_t { left, right } _side;

  explicit either(Side side) noexcept : _side(side) {}

  template <typename... Args>
  void _construct_left(Args&&... args) {
    new (&_left) Left(std::forward<Args>(
        args)...); // NOLINT(cppcoreguidelines-pro-type-union-access)
  }
  template <typename... Args>
  void _construct_right(Args&&... args) {
    new (&_right) Right(std::forward<Args>(
        args)...); // NOLINT(cppcoreguidelines-pro-type-union-access)
  }
  void _destruct() noexcept {
    if (_side == Side::left) {
      _left.~Left(); // NOLINT(cppcoreguidelines-pro-type-union-access)
    } else {
      _right.~Right(); // NOLINT(cppcoreguidelines-pro-type-union-access)
    }
  }

  template <typename Left_, typename Right_, typename... Args>
  friend either<Left_, Right_> make_left(Args&&... args);

  template <typename Left_, typename Right_, typename... Args>
  friend either<Left_, Right_> make_right(Args&&... args);
};

template <class Left, class Right>
inline bool operator==(
    const either<Left, Right>& lhs,
    const either<Left, Right>& rhs) {
  if (lhs.is_left() != rhs.is_left()) {
    return false;
  }
  if (lhs.is_left()) {
    return lhs.left() == rhs.left();
  } else {
    return lhs.right() == rhs.right();
  }
}

template <class Left, class Right>
inline bool operator!=(
    const either<Left, Right>& lhs,
    const either<Left, Right>& rhs) {
  return !operator==(lhs, rhs);
}

template <class Left, class Right>
inline std::ostream& operator<<(
    std::ostream& stream,
    const either<Left, Right>& value) {
  if (value.is_left()) {
    stream << "Left(" << value.left() << ")";
  } else {
    stream << "Right(" << value.right() << ")";
  }
  return stream;
}

template <typename Left, typename Right, typename... Args>
inline either<Left, Right> make_left(Args&&... args) {
  either<Left, Right> result(either<Left, Right>::Side::left);
  result._construct_left(std::forward<Args>(args)...);
  return result;
}

template <typename Left, typename Right, typename... Args>
inline either<Left, Right> make_right(Args&&... args) {
  either<Left, Right> result(either<Left, Right>::Side::right);
  result._construct_right(std::forward<Args>(args)...);
  return result;
}
} // namespace c10
