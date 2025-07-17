#pragma once

#include <c10/core/impl/PyInterpreterHooks.h>

namespace torch::detail {

// Concrete implementation of PyInterpreterHooks
class PyInterpreterHooks : public c10::impl::PyInterpreterHooksInterface {
 public:
  explicit PyInterpreterHooks(c10::impl::PyInterpreterHooksArgs);

  c10::impl::PyInterpreter* getPyInterpreter() const override;
};

} // namespace torch::detail
