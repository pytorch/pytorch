// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <memory>

#include <dlfcn.h>

#include "opentelemetry/plugin/detail/dynamic_library_handle.h"
#include "opentelemetry/plugin/detail/loader_info.h"
#include "opentelemetry/plugin/detail/utility.h"
#include "opentelemetry/plugin/factory.h"
#include "opentelemetry/plugin/hook.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace plugin
{
class DynamicLibraryHandleUnix final : public DynamicLibraryHandle
{
public:
  explicit DynamicLibraryHandleUnix(void *handle) noexcept : handle_{handle} {}

  ~DynamicLibraryHandleUnix() override { ::dlclose(handle_); }

private:
  void *handle_;
};

inline std::unique_ptr<Factory> LoadFactory(const char *plugin, std::string &error_message) noexcept
{
  dlerror();  // Clear any existing error.

  auto handle = ::dlopen(plugin, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr)
  {
    detail::CopyErrorMessage(dlerror(), error_message);
    return nullptr;
  }

  std::shared_ptr<DynamicLibraryHandle> library_handle{new (std::nothrow)
                                                           DynamicLibraryHandleUnix{handle}};
  if (library_handle == nullptr)
  {
    return nullptr;
  }

  auto make_factory_impl =
      reinterpret_cast<OpenTelemetryHook *>(::dlsym(handle, "OpenTelemetryMakeFactoryImpl"));
  if (make_factory_impl == nullptr)
  {
    detail::CopyErrorMessage(dlerror(), error_message);
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
