// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "opentelemetry/plugin/detail/dynamic_library_handle.h"
#include "opentelemetry/plugin/detail/loader_info.h"
#include "opentelemetry/plugin/detail/utility.h"
#include "opentelemetry/plugin/factory.h"
#include "opentelemetry/plugin/hook.h"
#include "opentelemetry/version.h"

#include <Windows.h>

#include <WinBase.h>
#include <errhandlingapi.h>

OPENTELEMETRY_BEGIN_NAMESPACE
namespace plugin
{
namespace detail
{
inline void GetLastErrorMessage(std::string &error_message) noexcept
{
  auto error_code = ::GetLastError();
  // See https://stackoverflow.com/a/455533/4447365
  LPTSTR error_text = nullptr;
  auto size         = ::FormatMessage(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPTSTR>(&error_text), 0, nullptr);
  if (size == 0)
  {
    return;
  }
  CopyErrorMessage(error_text, error_message);
  ::LocalFree(error_text);
}
}  // namespace detail

class DynamicLibraryHandleWindows final : public DynamicLibraryHandle
{
public:
  explicit DynamicLibraryHandleWindows(HINSTANCE handle) : handle_{handle} {}

  ~DynamicLibraryHandleWindows() override { ::FreeLibrary(handle_); }

private:
  HINSTANCE handle_;
};

inline std::unique_ptr<Factory> LoadFactory(const char *plugin, std::string &error_message) noexcept
{
  auto handle = ::LoadLibrary(plugin);
  if (handle == nullptr)
  {
    detail::GetLastErrorMessage(error_message);
    return nullptr;
  }

  std::shared_ptr<DynamicLibraryHandle> library_handle{new (std::nothrow)
                                                           DynamicLibraryHandleWindows{handle}};
  if (library_handle == nullptr)
  {
    detail::CopyErrorMessage("Allocation failure", error_message);
    return nullptr;
  }

  auto make_factory_impl = reinterpret_cast<OpenTelemetryHook *>(
      ::GetProcAddress(handle, "OpenTelemetryMakeFactoryImpl"));
  if (make_factory_impl == nullptr)
  {
    detail::GetLastErrorMessage(error_message);
    return nullptr;
  }
  if (*make_factory_impl == nullptr)
  {
    detail::CopyErrorMessage("Invalid plugin hook", error_message);
    return nullptr;
  }

  LoaderInfo loader_info;
  nostd::unique_ptr<char[]> plugin_error_message;
  auto factory_impl = (**make_factory_impl)(loader_info, plugin_error_message);
  if (factory_impl == nullptr)
  {
    detail::CopyErrorMessage(plugin_error_message.get(), error_message);
    return nullptr;
  }
  return std::unique_ptr<Factory>{new (std::nothrow)
                                      Factory{std::move(library_handle), std::move(factory_impl)}};
}
}  // namespace plugin
OPENTELEMETRY_END_NAMESPACE
