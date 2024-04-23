// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Make the entry point visible, loaded dynamically

#if defined(_MSC_VER)
// component_d is a DLL

#  ifdef BUILD_COMPONENT_D
__declspec(dllexport)
#  else
__declspec(dllimport)
#  endif

#else
// component_d is a shared library (*.so)
// component_d is compiled with visibility("hidden"),
__attribute__((visibility("default")))
#endif

    void do_something_in_d();
