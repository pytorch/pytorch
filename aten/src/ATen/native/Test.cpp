#include <ATen/Functions.h>

namespace at {
  namespace native {
    bool _test_expanding_array(ExpandingArray<3> ea, int64_t value) {
      assert(ea.size() == 3);
      IntList il = ea;
      for (int i = 0; i < il.size(); ++i) {
        assert(il[i] == value);
      }
      return true;
    }
  } // namespace native
} // namespace at
