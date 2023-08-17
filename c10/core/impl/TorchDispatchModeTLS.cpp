#include <c10/core/DispatchKey.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/irange.h>

#include <utility>

namespace c10 {
namespace impl {

thread_local TorchDispatchModeTLS torchDispatchModeState;

bool TorchDispatchModeTLS::any_modes_set(bool skip_infra_modes) {
  if (!torchDispatchModeState.stack_.empty())
    return true;
  if (!skip_infra_modes) {
    for (const auto i : c10::irange(
             static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
      if (torchDispatchModeState.infra_modes_[i] != c10::nullopt)
        return true;
    }
  }
  return false;
}

void TorchDispatchModeTLS::push_onto_stack(
    std::shared_ptr<SafePyObject> mode,
    c10::optional<TorchDispatchModeKey> mode_key) {
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
  if (mode_key == c10::nullopt) {
    torchDispatchModeState.stack_.push_back(std::move(mode));
  } else {
    set_mode(mode, mode_key.value());
  }
}

const std::
    tuple<std::shared_ptr<SafePyObject>, c10::optional<TorchDispatchModeKey>>
    TorchDispatchModeTLS::pop_stack(
        c10::optional<TorchDispatchModeKey> maybe_mode_key) {
  TORCH_CHECK(any_modes_set(), "trying to pop from empty mode stack");

  std::shared_ptr<SafePyObject> out;
  c10::optional<TorchDispatchModeKey> mode_key = c10::nullopt;

  if (maybe_mode_key.has_value()) {
    // Handle the case where we need to pop a *specific* mode off the stack,
    // based on the mode key.
    auto maybe_out =
        torchDispatchModeState
            .infra_modes_[static_cast<size_t>(maybe_mode_key.value())];
    TORCH_CHECK(
        maybe_out.has_value(),
        "trying to pop mode with mode_key=",
        to_string(mode_key.value()),
        "but that mode is not currently est");
    torchDispatchModeState
        .infra_modes_[static_cast<size_t>(maybe_mode_key.value())] =
        c10::nullopt;
    out = maybe_out.value();
    mode_key = maybe_mode_key;
  } else {
    // Handle the general case: pop the highest priority mode.
    if (!torchDispatchModeState.stack_.empty()) {
      // First try to grab a mode from the stack
      out = torchDispatchModeState.stack_.back();
      torchDispatchModeState.stack_.pop_back();
    } else {
      // Otherwise, try to grab a mode from infra_modes_
      for (const auto i : c10::irange(
               static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
        if (torchDispatchModeState.infra_modes_[i] != c10::nullopt) {
          out = torchDispatchModeState.infra_modes_[i].value();
          mode_key = static_cast<TorchDispatchModeKey>(i);
          torchDispatchModeState.infra_modes_[i] = c10::nullopt;
          break;
        }
      }
    }
    TORCH_INTERNAL_ASSERT(out != nullptr);
  }

  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return std::make_tuple(out, mode_key);
}

const std::shared_ptr<SafePyObject>& TorchDispatchModeTLS::get_stack_at(
    int64_t idx) {
  TORCH_CHECK(idx < stack_len(), "Tried to get stack at idx that's too big");
  // Our "logical" stack includes both:
  // - any user modes (the entire torchDispatchModeState.stack_)
  // - any infra modes (members of torchDispatchModeState.infra_modes_ that are
  // not None)

  // idx == 0 means the "bottom" of the stack, which starts with any infra
  // modes.
  auto curr_idx = idx;
  for (const auto i :
       c10::irange(static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
    if (torchDispatchModeState.infra_modes_[i] != c10::nullopt) {
      if (curr_idx == 0) {
        return torchDispatchModeState.infra_modes_[i].value();
      }
      curr_idx -= 1;
    }
  }
  // At this point, we're guaranteed that curr_idx < stack_.size()
  return torchDispatchModeState.stack_[curr_idx];
}

int64_t TorchDispatchModeTLS::stack_len() {
  auto stack_len = static_cast<int64_t>(torchDispatchModeState.stack_.size());
  int64_t infra_modes_len = 0;
  for (const auto i :
       c10::irange(static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
    if (torchDispatchModeState.infra_modes_[i] != c10::nullopt) {
      infra_modes_len += 1;
    }
  }
  return stack_len + infra_modes_len;
}

const c10::optional<std::shared_ptr<SafePyObject>> TorchDispatchModeTLS::
    get_mode(TorchDispatchModeKey mode_key) {
  return torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)];
}

void TorchDispatchModeTLS::set_mode(
    const std::shared_ptr<SafePyObject>& mode,
    TorchDispatchModeKey mode_key) {
  TORCH_CHECK(
      torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] ==
          c10::nullopt,
      "trying to set the current ",
      to_string(mode_key),
      ", but one already exists");

  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }

  torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] = mode;
}

const c10::optional<std::shared_ptr<SafePyObject>> TorchDispatchModeTLS::
    unset_mode(TorchDispatchModeKey mode_key) {
  auto out = torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)];
  torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] =
      c10::nullopt;
  if (out.has_value() && !any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return out;
}

const TorchDispatchModeTLS& TorchDispatchModeTLS::get_state() {
  return torchDispatchModeState;
}

void TorchDispatchModeTLS::set_state(TorchDispatchModeTLS state) {
  torchDispatchModeState = std::move(state);
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  } else {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
}

// UTIL

bool dispatch_mode_enabled(bool skip_infra_modes) {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python) &&
      TorchDispatchModeTLS::any_modes_set(
          /*skip_infra_modes=*/skip_infra_modes);
}

std::string to_string(TorchDispatchModeKey mode_key) {
  switch (mode_key) {
    case TorchDispatchModeKey::PROXY:
      return "ProxyTorchDispatchMode";
    case TorchDispatchModeKey::FAKE:
      return "FakeTensorMode";
    default:
      return "UNKNOWN_MODE";
  }
}

} // namespace impl
} // namespace c10
