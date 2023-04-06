#pragma once

namespace c10::impl::cow {

constexpr auto enable_instrumentation() -> bool {
#if defined(PYTORCH_INSTRUMENT_COW_TENSOR)
  return true;
#else
  return false;
#endif
}

} // namespace c10::impl::cow
