#pragma once

namespace at {

  template<typename To, typename From>
  static inline To HalfFix(From h) {
    return To { h.x };
  }

}
