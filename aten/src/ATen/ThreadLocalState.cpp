#include <ATen/ThreadLocalState.h>

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/grad_mode.h>
#endif

namespace at {

namespace {
thread_local bool is_record_function_enabled_ = true;
typedef std::array<std::function<SettingValue()>, (size_t)ThreadLocalSetting::NUM_SETTINGS> getters_arr;
getters_arr& getters() {
  static getters_arr getters;
  return getters;
}

typedef std::array<std::function<void(SettingValue)>, (size_t)ThreadLocalSetting::NUM_SETTINGS> setters_arr;
setters_arr& setters() {
  static setters_arr setters;
  return setters;
}

bool _unused = []() {
  ThreadLocalState::registerThreadLocalSetting(
    ThreadLocalSetting::GRAD_MODE,
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
    []() {
      auto v = SettingValue();
      v.value = GradMode::is_enabled();
      return v;
    },
    [](SettingValue v) {
      GradMode::set_enabled(v.value);
    }
#else
    []() { return SettingValue{.value = false}; },
    [](SettingValue v) {}
#endif
  );

  ThreadLocalState::registerThreadLocalSetting(
    ThreadLocalSetting::RECORD_FUNCTION,
    []() {
      auto v = SettingValue();
      v.value = _tls_is_record_function_enabled();
      return v;
    },
    [](SettingValue v) {
      _tls_set_record_function_enabled(v.value);
    }
  );
  return true;
}();

} // namespace

ThreadLocalState::ThreadLocalState(bool keep_grad_mode)
    : dispatch_key_(c10::impl::tls_local_dispatch_key_set()),
      debug_info_(ThreadLocalDebugInfo::_current()),
      keep_grad_mode_(keep_grad_mode) {
  for (auto st = (size_t)0; st < (size_t)ThreadLocalSetting::NUM_SETTINGS; ++st) {
    if (!getters()[st] ||
        (st == (size_t)ThreadLocalSetting::GRAD_MODE && !keep_grad_mode_)) {
      continue;
    }
    settings_[st] = getters()[st]();
  }
}

/* static */
void ThreadLocalState::setThreadLocalState(
    const ThreadLocalState& state) {
for (auto st = (size_t)0; st < (size_t)ThreadLocalSetting::NUM_SETTINGS; ++st) {
    if (!setters()[st] ||
        (st == (size_t)ThreadLocalSetting::GRAD_MODE && !state.keep_grad_mode_)) {
      continue;
    }
    setters()[st](state.settings_[st]);
  }

  c10::impl::_force_tls_local_dispatch_key_set(state.dispatch_key_);

  ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_);
}

/* static */
void ThreadLocalState::registerThreadLocalSetting(
      ThreadLocalSetting st,
      std::function<SettingValue(void)> getter,
      std::function<void(SettingValue)> setter) {
  auto st_ = (size_t)st;
  TORCH_CHECK(!getters()[st_] && !setters()[st_],
      "Setting with the key ", st_, " is already registered");
  TORCH_CHECK(getter && setter, "Expected non empty getter/setter");
  getters()[st_] = std::move(getter);
  setters()[st_] = std::move(setter);
}

bool _tls_is_record_function_enabled() {
  return is_record_function_enabled_;
}
void _tls_set_record_function_enabled(bool is_enabled) {
  is_record_function_enabled_ = is_enabled;
}

} // namespace at
