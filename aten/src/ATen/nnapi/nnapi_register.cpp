// Set flag if running on ios
#ifdef __APPLE__
  #include <TargetConditionals.h>
  #if TARGET_OS_IPHONE
    #define IS_IOS_NNAPI_BIND
  #endif
#endif

#ifndef IS_IOS_NNAPI_BIND
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto register_NnapiCompilation = [](){
  try {
    TORCH_LIBRARY("_nnapi", m) {
        register_nnapi(m);
    }
  } catch (std::exception& exn) {
    LOG(ERROR) << "Failed to register class nnapi.Compilation: " << exn.what();
    throw;
  }
}();
#else
  #undef IS_IOS_NNAPI_BIND
#endif
