// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Make the entry point visible, loaded dynamically

#if defined(_MSC_VER)
// component_f is a DLL

#  ifdef BUILD_COMPONENT_F
__declspec(dllexport)
#  else
__declspec(dllimport)
#  endif

#else
// component_f is a shared library (*.so)
// component_f is compiled with visibility("hidden"),
__attribute__((visibility("default")))
#endif

    void do_something_in_f();
