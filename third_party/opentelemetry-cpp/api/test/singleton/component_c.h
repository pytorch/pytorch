// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#if defined(_MSC_VER)
// component_c is a DLL

#  ifdef BUILD_COMPONENT_C
__declspec(dllexport)
#  else
__declspec(dllimport)
#  endif

#endif

    void do_something_in_c();
