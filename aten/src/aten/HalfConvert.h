#pragma once

namespace tlib {

  template<typename To, typename From>
  static inline To HalfFix(From h) {
    return To { h.x };
  }

}
