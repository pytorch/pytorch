/*
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import weakref
import warnings


class RemovableHandle(object):
    """A handle which provides the capability to remove a hook."""

    next_id = 0

    def __init__(self, hooks_dict):
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self):
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __getstate__(self):
        return (self.hooks_dict_ref(), self.id)

    def __setstate__(self, state):
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(collections.OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()
*/

#pragma once

#include <torch/csrc/autograd/cpp_hook.h>

namespace torch {
namespace utils {
namespace hooks {

/// A handle which provides the capability to remove a hook.
class RemovableHandle {
 public:
  // yf225 TODO: let's use a std::weak_ptr to mimick Python version behavior
  explicit RemovableHandle(torch::autograd::hooks_map hooks_map) {}
 private:
  static int64_t next_id;
};

} // namespace hooks
} // namespace utils
} // namespace torch