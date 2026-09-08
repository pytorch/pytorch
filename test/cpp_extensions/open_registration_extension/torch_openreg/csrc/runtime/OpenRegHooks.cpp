#include "OpenRegHooks.h"

// LITERALINCLUDE START: OPENREG HOOK REGISTER
namespace c10::openreg {

static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());

  return true;
}();

} // namespace c10::openreg
// LITERALINCLUDE END: OPENREG HOOK REGISTER

namespace {

struct IpcCloseCtx {
  void* ptr;
  size_t size;
};

} // namespace

namespace c10::openreg {

at::IpcMemHandle OpenRegHooksInterface::getIpcMemHandle(void* ptr) const {
#ifndef _WIN32
  char name[OR_IPC_HANDLE_MAX_LEN];
  ptrdiff_t offset = 0;
  orError_t err = orGetIpcMemHandle(ptr, name, sizeof(name), &offset);
  TORCH_CHECK(err == orSuccess, "orGetIpcMemHandle failed (err=", err, ")");
  return at::IpcMemHandle{offset, std::string(name)};
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "OpenReg IPC is not supported on Windows.");
#endif
}

c10::DataPtr OpenRegHooksInterface::openIpcMemHandle(
    const std::string& handle) const {
#ifndef _WIN32
  void* ptr = nullptr;
  size_t size = 0;
  orError_t err = orOpenIpcMemHandle(&ptr, handle.c_str(), &size);
  TORCH_CHECK(err == orSuccess, "orOpenIpcMemHandle failed (err=", err, ")");

  auto* ctx = new IpcCloseCtx{ptr, size};
  return c10::DataPtr(
      ptr,
      ctx,
      [](void* raw_ctx) {
        auto* c = static_cast<IpcCloseCtx*>(raw_ctx);
        orCloseIpcMemHandle(c->ptr, c->size);
        delete c;
      },
      at::Device(at::DeviceType::PrivateUse1, current_device()));
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "OpenReg IPC is not supported on Windows.");
#endif
}

} // namespace c10::openreg
